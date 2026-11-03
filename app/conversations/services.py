"""
Conversation service layer.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.models import AgentConfig
from app.conversations.models import Conversation, Message
from app.integrations.google.calendar import GoogleCalendarService
from app.integrations.whatsapp.client import WhatsAppClient
from app.llm_providers.base import ConversationMessage
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Orchestrates the conversation loop:
    Incoming WhatsApp Message -> History Lookup -> LLM Call -> WhatsApp Reply -> Save to DB.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.whatsapp_client = WhatsAppClient()
        self.llm_factory = LLMProviderFactory()

    async def _get_or_create_conversation(
        self, organization_id: str, customer_phone: str, customer_name: str | None
    ) -> Conversation:
        """Fetch active conversation for a customer, or create a new one."""
        stmt = (
            select(Conversation)
            .where(
                Conversation.organization_id == organization_id,
                Conversation.customer_phone == customer_phone,
                Conversation.status == "active",
            )
            .options(selectinload(Conversation.messages))
        )
        result = await self.db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            conversation = Conversation(
                id=uuid.uuid4(),
                organization_id=organization_id,
                customer_phone=customer_phone,
                customer_name=customer_name or "Cliente",
                status="active",
            )
            self.db.add(conversation)
            await self.db.flush()
            # Initialize empty messages list for the LLM
            conversation.messages = []

        return conversation

    async def process_incoming_message(
        self,
        phone_number_id: str,
        customer_phone: str,
        text_content: str,
        customer_name: str | None,
    ) -> None:
        """
        Main entrypoint for processing an incoming WhatsApp message.
        """
        clean_phone_id = str(phone_number_id).strip()
        logger.info(
            f"Processing incoming WhatsApp message from {customer_phone} (phone_id={clean_phone_id}): '{text_content}'"
        )

        # 1. Look up AgentConfig by whatsapp_phone_number_id
        stmt = select(AgentConfig).where(AgentConfig.whatsapp_phone_number_id == clean_phone_id)
        result = await self.db.execute(stmt)
        agent_config = result.scalar_one_or_none()

        if not agent_config:
            # Fallback: find the active AgentConfig and associate it with this phone_number_id
            stmt_fallback = (
                select(AgentConfig)
                .order_by(AgentConfig.updated_at.desc())
                .limit(1)
            )
            result = await self.db.execute(stmt_fallback)
            agent_config = result.scalar_one_or_none()
            if agent_config:
                agent_config.whatsapp_phone_number_id = clean_phone_id
                agent_config.status = "active"
                self.db.add(agent_config)
                await self.db.commit()
                logger.info(
                    f"Associated AgentConfig {agent_config.id} with WhatsApp phone_number_id {clean_phone_id}"
                )

        if not agent_config:
            logger.warning(f"No AgentConfig found in database to handle phone_number_id: {clean_phone_id}")
            return

        organization_id = str(agent_config.organization_id)

        # Load Organization business name
        org_name = "tu negocio"
        try:
            org_res = await self.db.execute(
                select(Organization).where(Organization.id == agent_config.organization_id)
            )
            org = org_res.scalar_one_or_none()
            if org and org.name:
                org_name = org.name
        except Exception as org_err:
            logger.warning(f"Could not load organization name: {org_err}")

        # 2. Get or create conversation history
        effective_customer_name = (customer_name or "Cliente").strip()
        conversation = await self._get_or_create_conversation(
            organization_id, customer_phone, effective_customer_name
        )

        # 3. Append incoming user message
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            organization_id=organization_id,
            role="user",
            content=text_content,
        )
        self.db.add(user_message)
        conversation.messages.append(user_message)
        await self.db.commit()

        # 4. Prepare message history for LLM
        history_msgs = [
            ConversationMessage(role=m.role, content=m.content.strip())
            for m in conversation.messages[-8:]
            if m.content and m.content.strip()
        ]

        # 5. Build system prompt
        ans = dict((agent_config.onboarding_answers or {}) or {})
        agent_name = ans.get("agentName") or ans.get("agent_name") or "Asistente"
        services_text = ans.get("services") or "Asesoría y atención especializada"
        schedule_text = ans.get("schedule") or "Lunes a Viernes 8:00 AM - 6:00 PM"

        now_co = datetime.utcnow() - timedelta(hours=5)
        tomorrow = (now_co + timedelta(days=1)).strftime("%Y-%m-%d")
        default_avail = f"Atendemos {schedule_text}. Horarios recomendados: 10:00 AM y 3:30 PM."
        avail_info = default_avail

        cal_service = GoogleCalendarService(self.db, organization_id)
        lower_msg = text_content.lower()
        needs_calendar = any(
            kw in lower_msg
            for kw in ("cita", "hora", "agendar", "disponib", "espacio", "turno", "reservar", "mañana", "calendario", "cupo")
        )
        if needs_calendar:
            try:
                avail_res = await asyncio.wait_for(
                    cal_service.get_availability(tomorrow),
                    timeout=2.0,
                )
                res_str = str(avail_res)
                if res_str and "no disponible" not in res_str.lower() and "failed" not in res_str.lower():
                    avail_info = res_str
            except Exception as e:
                logger.info(f"Calendar availability check skipped/failed: {e}")

        base_prompt = agent_config.system_prompt_generated or f"Eres {agent_name}, asistente virtual de {org_name}."
        system_prompt = (
            f"{base_prompt}\n\n"
            "--- CONTEXTO EN TIEMPO REAL ---\n"
            f"Negocio: {org_name}\n"
            f"Fecha y hora actual: {now_co.strftime('%Y-%m-%d %H:%M')} (hora Colombia)\n"
            f"Cliente con quien hablas: {effective_customer_name}\n"
            f"Teléfono del cliente: {customer_phone}\n"
            f"Servicios: {services_text}\n"
            f"Horario de atención: {schedule_text}\n"
            f"Disponibilidad de citas ({tomorrow}): {avail_info}\n\n"
            "REGLAS OBLIGATORIAS DE RESPUESTA:\n"
            "- Hablas directamente con un cliente por WhatsApp. Sé siempre muy amable, cálido, conversacional y servicial.\n"
            "- Termina siempre tus oraciones e ideas de forma 100% completa y coherente (máximo 2 a 3 frases breves). NUNCA cortes palabras a la mitad.\n"
            "- Si el cliente saluda (ej: 'hola', 'buenas'), salúdalo amablemente, preséntate brevemente y pregúntale en qué le puedes colaborar hoy.\n"
            "- Si el cliente consulta disponibilidad o pide cita o reunión, dale opciones claras de inmediato (ej: 'Tenemos disponibilidad a las 10:00 AM o a las 3:30 PM, ¿cuál te queda mejor?').\n"
            f"- Ya conoces el nombre del cliente ({effective_customer_name}). Al agendar, NO le vuelvas a pedir su nombre ni su correo, confírmale directamente que la cita o reunión virtual ha quedado agendada.\n"
            "- OBLIGATORIO AGENDAR: Cuando el cliente elija o confirme una fecha u hora (ej: 'el jueves a las 11 am', 'a las 10 am', 'agenda por favor'), "
            "confirma de inmediato con entusiasmo que su cita o reunión ha quedado reservada y coloca SIEMPRE al final de tu mensaje el comando: "
            f"[[AGENDAR|YYYY-MM-DD|HH:MM|{effective_customer_name}|Servicio]]\n"
            "- OBLIGATORIO CANCELAR: Si el cliente solicita cancelar o anular su cita (ej: 'cancela mi cita', 'ya no puedo ir', 'anula la cita', 'deseo cancelar'), "
            "confirma con amabilidad y calidez que su cita ha sido cancelada y coloca SIEMPRE al final el comando: "
            f"[[CANCELAR|{effective_customer_name}]]\n"
            "- OBLIGATORIO REAGENDAR: Si el cliente solicita reprogramar o cambiar su cita (ej: 'cámbiame la cita a las 4pm', 'mejor el viernes a las 11'), "
            "confirma con amabilidad el cambio de horario y coloca SIEMPRE al final el comando: "
            f"[[REAGENDAR|YYYY-MM-DD|HH:MM|{effective_customer_name}|Servicio]]\n"
        )

        # 6. Call LLM with resilient candidates
        candidates = [
            ("google", "gemini-3.8-flash"),
            ("google", "gemini-flash-latest"),
            ("openai", "gpt-4o-mini"),
        ]
        reply_text = ""
        for pname, pmodel in candidates:
            try:
                provider = self.llm_factory.get_provider(pname, pmodel, system_prompt)
                resp = await asyncio.wait_for(
                    provider.generate(
                        system_prompt=system_prompt,
                        messages=history_msgs,
                        temperature=0.3,
                        max_tokens=500,
                    ),
                    timeout=12.0,
                )
                if resp.content and resp.content.strip():
                    reply_text = resp.content.strip()
                    break
            except Exception as e:
                logger.warning(f"LLM candidate {pname}/{pmodel} failed in WhatsApp processing: {e}")

        if not reply_text:
            reply_text = f"¡Hola {effective_customer_name}! Gracias por comunicarte con {org_name}. ¿En qué podemos colaborarte hoy?"

        # 7. Check for commands: CANCELAR, REAGENDAR, AGENDAR
        cancel_match = re.search(r"\[\[CANCELAR(?:\|([^\]]+))?\]\]", reply_text, re.IGNORECASE)
        if cancel_match:
            reply_text = re.sub(r"\[\[CANCELAR[^\]]*\]\]", "", reply_text).strip()
            try:
                evs = await cal_service.find_events(effective_customer_name)
                for ev in evs:
                    if ev.get("id"):
                        await cal_service.delete_appointment(ev["id"])
                        logger.info(f"Cancelled event {ev['id']} on Google Calendar")
                        break
            except Exception as e:
                logger.warning(f"Error deleting event on Google Calendar: {e}")

        m_reagendar = re.search(r"\[\[REAGENDAR\|([^|\]]+)\|([^|\]]+)\|([^|\]]*)\|([^\]]*)\]\]", reply_text, re.IGNORECASE)
        m_agendar = re.search(r"\[\[AGENDAR\|([^|\]]+)\|([^|\]]+)\|([^|\]]*)\|([^\]]*)\]\]", reply_text, re.IGNORECASE)

        if m_reagendar or m_agendar:
            match = m_reagendar or m_agendar
            b_date, b_time, b_name, b_service = (g.strip() for g in match.groups())
            if m_reagendar:
                reply_text = re.sub(r"\[\[REAGENDAR[^\]]*\]\]", "", reply_text).strip()
                try:
                    evs = await cal_service.find_events(effective_customer_name)
                    for ev in evs:
                        if ev.get("id"):
                            await cal_service.delete_appointment(ev["id"])
                            break
                except Exception:
                    pass
            else:
                reply_text = re.sub(r"\[\[AGENDAR[^\]]*\]\]", "", reply_text).strip()

            # Schedule new appointment on Google Calendar
            try:
                parts = b_time.split(":")
                hh = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 10
                mm = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                start_iso = f"{b_date}T{hh:02d}:{mm:02d}:00"
                end_iso = f"{b_date}T{(hh+1):02d}:{mm:02d}:00"
                await cal_service.create_appointment(
                    customer_name=b_name or effective_customer_name,
                    customer_phone=customer_phone,
                    start_time=start_iso,
                    end_time=end_iso,
                )
                logger.info(f"Booked appointment on Google Calendar for {b_name} at {start_iso}")
            except Exception as e:
                logger.warning(f"Error booking Google Calendar appointment: {e}")

        # Final cleanup of any stray tags
        reply_text = re.sub(r"\[\[[A-Z_]+[^\]]*\]\]", "", reply_text).strip()

        # 8. Save assistant response to DB
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            organization_id=organization_id,
            role="assistant",
            content=reply_text,
        )
        self.db.add(assistant_message)
        await self.db.commit()

        # 9. Send via WhatsApp
        logger.info(f"Sending WhatsApp reply to {customer_phone}: '{reply_text}'")
        try:
            await self.whatsapp_client.send_text_message(
                phone_number_id=clean_phone_id,
                to=customer_phone,
                text=reply_text,
            )
            logger.info(f"WhatsApp reply successfully sent to {customer_phone}")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message to {customer_phone}: {e}")

    async def list_conversations(
        self, organization_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Conversation], int]:
        """
        List conversations for an organization with pagination and their latest message.
        """
        from sqlalchemy import func

        offset = (page - 1) * page_size

        # Count total
        count_stmt = select(func.count(Conversation.id)).where(
            Conversation.organization_id == organization_id
        )
        count_res = await self.db.execute(count_stmt)
        total = count_res.scalar_one()

        # Fetch page with eager loaded messages
        stmt = (
            select(Conversation)
            .where(Conversation.organization_id == organization_id)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.db.execute(stmt)
        conversations = list(result.scalars().all())

        return conversations, total

    async def get_conversation(
        self, organization_id: str, conversation_id: str
    ) -> Conversation | None:
        """
        Retrieve a specific conversation with all its messages.
        """
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.organization_id == organization_id,
            )
            .options(selectinload(Conversation.messages))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
