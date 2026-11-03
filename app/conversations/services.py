"""
Conversation service layer.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import re
import uuid

from sqlalchemy import select, text
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
        self, organization_id: str | uuid.UUID, customer_phone: str, customer_name: str | None
    ) -> Conversation:
        """Fetch the latest conversation for a customer in this organization, or create a new one."""
        org_u = uuid.UUID(str(organization_id)) if not isinstance(organization_id, uuid.UUID) else organization_id
        clean_phone = "".join(filter(str.isdigit, str(customer_phone)))
        phone_variants = [customer_phone, clean_phone, f"+{clean_phone}"]

        # 1. Search for existing conversation for this customer under this organization
        stmt = (
            select(Conversation)
            .where(
                Conversation.organization_id == org_u,
                Conversation.customer_phone.in_(phone_variants),
            )
            .order_by(Conversation.created_at.desc())
        )
        result = await self.db.execute(stmt)
        conversation = result.scalars().first()

        # 2. If not found in this org, check if customer already had a conversation in any org to reuse
        if not conversation:
            stmt_any = (
                select(Conversation)
                .where(Conversation.customer_phone.in_(phone_variants))
                .order_by(Conversation.created_at.desc())
            )
            res_any = await self.db.execute(stmt_any)
            existing_conv = res_any.scalars().first()
            if existing_conv:
                existing_conv.organization_id = org_u
                existing_conv.status = "active"
                if customer_name and customer_name != "Cliente":
                    existing_conv.customer_name = customer_name
                self.db.add(existing_conv)
                await self.db.commit()
                conversation = existing_conv

        # 3. Only if absolutely no prior conversation exists for this customer, create one
        if not conversation:
            conversation = Conversation(
                id=uuid.uuid4(),
                organization_id=org_u,
                customer_phone=customer_phone,
                customer_name=customer_name or "Cliente",
                status="active",
            )
            self.db.add(conversation)
            await self.db.commit()

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

        # 1. Look up AgentConfig by setting tenant context on each organization
        # (Organizations has NO RLS, so it safely discovers the tenant without RLS violation)
        agent_config = None
        orgs_res = await self.db.execute(select(Organization).order_by(Organization.created_at.desc()))
        orgs = list(orgs_res.scalars().all())

        for org in orgs:
            await self.db.execute(
                text("SELECT set_config('app.current_organization_id', :org_id, false)"),
                {"org_id": str(org.id)},
            )
            stmt_cfg = select(AgentConfig).where(AgentConfig.organization_id == org.id)
            cfg_res = await self.db.execute(stmt_cfg)
            cfg = cfg_res.scalar_one_or_none()
            if cfg:
                if cfg.whatsapp_phone_number_id == clean_phone_id:
                    agent_config = cfg
                    break
                elif not agent_config:
                    agent_config = cfg

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

        # Set RLS context for this tenant for all subsequent queries (Conversation, Message, Calendar)
        await self.db.execute(
            text("SELECT set_config('app.current_organization_id', :org_id, false)"),
            {"org_id": organization_id},
        )

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
        await self.db.commit()

        # 4. Prepare message history for LLM in natural chronological order (clean up trailing broken syntax)
        stmt_history = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
        )
        hist_res = await self.db.execute(stmt_history)
        all_msgs = list(hist_res.scalars().all())
        recent_msgs = all_msgs[-12:]

        history_msgs = []
        for m in recent_msgs:
            c = (m.content or "").strip()
            if not c:
                continue
            if m.role in ("assistant", "model"):
                c = re.sub(r"\[\[.*?\]\]", "", c).strip()
                c = re.sub(r"[\s,;:-]+$", "", c).strip()
                if not c.endswith((".", "!", "?")):
                    c += "."
            history_msgs.append(ConversationMessage(role=m.role, content=c))

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
            f"Horario de atención: {schedule_text}\n"
            f"Disponibilidad de citas ({tomorrow}): {avail_info}\n\n"
            "REGLAS ESTRICTAS DE RESPUESTA EN WHATSAPP:\n"
            "1. BREVEDAD: Responde siempre en máximo 1 o 2 oraciones breves, cálidas y directas. NUNCA envíes párrafos largos ni repitas descripciones corporativas o listas de servicios completas.\n"
            "2. ORACIONES COMPLETAS: Cada oración debe tener sentido gramatical completo y terminar con su punto final (ej: '¡Listo, Andrés! Tu cita ha quedado agendada para mañana a las 10:30 AM. ¡Te esperamos!'). NUNCA dejes frases cortadas ni incompletas.\n"
            "3. COMANDOS DE ACCIÓN (siempre en una línea nueva al final del mensaje):\n"
            f"   - Agendar cita: Confirma la hora completa en el texto y al final pon en renglón aparte: [[AGENDAR|{tomorrow}|HH:MM]]\n"
            "   - Cancelar: Confirma la cancelación en el texto y al final pon en renglón aparte: [[CANCELAR]]\n"
            "   - Reagendar: Confirma el nuevo horario en el texto y al final pon en renglón aparte: [[REAGENDAR|YYYY-MM-DD|HH:MM]]\n"
            "   - Agradecimiento o despedida: Responde con amabilidad en 1 frase (ej: '¡Con mucho gusto! Feliz día.') SIN incluir ningún comando ni tag.\n"
        )

        # 6. Call LLM with resilient candidates
        candidates = [
            ("google", "gemini-flash-latest"),
            ("google", "gemini-1.5-flash"),
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
                        max_tokens=250,
                    ),
                    timeout=12.0,
                )
                if resp.content and resp.content.strip():
                    reply_text = resp.content.strip()
                    break
            except Exception as e:
                logger.warning(f"LLM candidate {pname}/{pmodel} failed in WhatsApp processing: {e}", exc_info=True)

        if not reply_text:
            reply_text = f"¡Hola {effective_customer_name}! Gracias por comunicarte con {org_name}. ¿En qué podemos colaborarte hoy?"

        # 7. Check for commands: CANCELAR, REAGENDAR, AGENDAR
        cmd_match = re.search(r"\[\[\s*(AGENDAR|REAGENDAR|CANCELAR)\b([^\]]*)\]\]", reply_text, re.IGNORECASE)
        command_type = None
        cmd_args = []
        if cmd_match:
            command_type = cmd_match.group(1).upper()
            raw_args = cmd_match.group(2).strip()
            if raw_args.startswith(("|", ":")):
                raw_args = raw_args[1:].strip()
            cmd_args = [p.strip() for p in raw_args.split("|") if p.strip()]

        lower_user = text_content.lower()
        # Failsafe: if user requested scheduling and specified time, but LLM tag was missing/malformed:
        if not command_type and any(kw in lower_user for kw in ("agendar", "cita", "reservar", "programar")):
            time_m = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)\b", lower_user, re.IGNORECASE)
            if time_m:
                command_type = "AGENDAR"
                cmd_args = [tomorrow, time_m.group(1)]
                logger.info(f"Failsafe intent detected AGENDAR for time: {time_m.group(1)}")

        hh, mm = 10, 30
        if command_type == "CANCELAR":
            try:
                evs = await cal_service.find_events(effective_customer_name)
                for ev in evs:
                    if ev.get("id"):
                        await cal_service.delete_appointment(ev["id"])
                        logger.info(f"Cancelled event {ev['id']} on Google Calendar")
                        break
            except Exception as e:
                logger.warning(f"Error deleting event on Google Calendar: {e}")

        elif command_type in ("AGENDAR", "REAGENDAR"):
            if command_type == "REAGENDAR":
                try:
                    evs = await cal_service.find_events(effective_customer_name)
                    for ev in evs:
                        if ev.get("id"):
                            await cal_service.delete_appointment(ev["id"])
                            break
                except Exception:
                    pass

            # Extract target date
            b_date = tomorrow
            if len(cmd_args) > 0:
                dm = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", cmd_args[0])
                if dm:
                    b_date = dm.group(1)

            # Extract time string
            raw_time_str = ""
            if len(cmd_args) > 1:
                raw_time_str = cmd_args[1]
            elif len(cmd_args) == 1 and not re.search(r"^\d{4}-\d{2}-\d{2}$", cmd_args[0]):
                raw_time_str = cmd_args[0]
            if not raw_time_str:
                tm = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)\b", lower_user, re.IGNORECASE)
                raw_time_str = tm.group(1) if tm else "10:30 AM"

            # Parse hours & minutes accurately (handles '10:30 am', '10:30', '10am', '4pm')
            raw_t_lower = raw_time_str.lower().strip()
            is_pm = "pm" in raw_t_lower or "p.m." in raw_t_lower or "tarde" in raw_t_lower or "noche" in raw_t_lower
            is_am = "am" in raw_t_lower or "a.m." in raw_t_lower
            digits_colon = re.sub(r"[^\d:]", "", raw_t_lower)
            if ":" in digits_colon:
                t_parts = digits_colon.split(":")
                hh = int(t_parts[0]) if t_parts[0].isdigit() else 10
                mm = int(t_parts[1]) if len(t_parts) > 1 and t_parts[1].isdigit() else 0
            elif digits_colon.isdigit():
                hh = int(digits_colon)
                mm = 0
            if is_pm and hh < 12:
                hh += 12
            elif is_am and hh == 12:
                hh = 0

            b_name = cmd_args[2] if len(cmd_args) > 2 else effective_customer_name
            b_service = cmd_args[3] if len(cmd_args) > 3 else "Asesoría especializada"

            start_iso = f"{b_date}T{hh:02d}:{mm:02d}:00-05:00"
            end_iso = f"{b_date}T{(hh+1):02d}:{mm:02d}:00-05:00"

            # Prevent duplicate bookings for the same customer at the same slot
            already_booked = False
            try:
                evs = await cal_service.find_events(b_name)
                slot_prefix = start_iso[:16]
                already_booked = any(
                    ev.get("start", {}).get("dateTime", "").startswith(slot_prefix)
                    for ev in evs
                )
            except Exception as ev_err:
                logger.info(f"Could not check existing calendar events: {ev_err}")

            if already_booked:
                logger.info(f"Event for {b_name} at {start_iso[:16]} already exists in Google Calendar. Skipping duplicate creation.")
            else:
                try:
                    cal_res = await cal_service.create_appointment(
                        customer_name=b_name,
                        customer_phone=customer_phone,
                        start_time=start_iso,
                        end_time=end_iso,
                        notes=b_service,
                    )
                    logger.info(f"Booked appointment on Google Calendar for {b_name} at {start_iso}: {cal_res}")
                except Exception as e:
                    logger.warning(f"Error booking Google Calendar appointment: {e}")

        # Clean tags from text - ONLY strip double brackets [[...]], NEVER single brackets
        reply_text = re.sub(r"\[\[\s*[A-Z_]+[^\]]*\]\]", "", reply_text).strip()
        reply_text = re.sub(r"\[\[.*?\]\]", "", reply_text).strip()
        reply_text = re.sub(r"\[\[.*$", "", reply_text).strip()

        # Fix truncated sentences ending in incomplete prepositions or punctuation
        reply_text = re.sub(r"[\s,;:-]+$", "", reply_text).strip()
        incomplete_pattern = r"\b(para mañana|mañana|a las|a|el día|para|las|el|en|confirmada para|quedó agendada para|quedó para)$"
        if re.search(incomplete_pattern, reply_text, re.IGNORECASE):
            hh_12 = ((hh - 1) % 12) + 1
            ampm = "PM" if hh >= 12 else "AM"
            if re.search(r"\b(a las|a)$", reply_text, re.IGNORECASE):
                reply_text = re.sub(r"\b(a las|a)$", f"a las {hh_12}:{mm:02d} {ampm}.", reply_text)
            else:
                reply_text = f"{reply_text} a las {hh_12}:{mm:02d} {ampm}."
        elif not reply_text.endswith((".", "!", "?")):
            reply_text = reply_text + "."

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
        self, organization_id: str | uuid.UUID, page: int = 1, page_size: int = 20
    ) -> tuple[list[Conversation], int]:
        """
        List conversations for an organization with pagination and their latest message.
        """
        from sqlalchemy import func

        org_u = uuid.UUID(str(organization_id)) if not isinstance(organization_id, uuid.UUID) else organization_id
        offset = (page - 1) * page_size

        # Count total
        count_stmt = select(func.count(Conversation.id)).where(
            Conversation.organization_id == org_u
        )
        count_res = await self.db.execute(count_stmt)
        total = count_res.scalar_one()

        # Fetch page with eager loaded messages
        stmt = (
            select(Conversation)
            .where(Conversation.organization_id == org_u)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.db.execute(stmt)
        conversations = list(result.scalars().all())

        return conversations, total

    async def get_conversation(
        self, organization_id: str | uuid.UUID, conversation_id: str | uuid.UUID
    ) -> Conversation | None:
        """
        Retrieve a specific conversation with all its messages.
        """
        org_u = uuid.UUID(str(organization_id)) if not isinstance(organization_id, uuid.UUID) else organization_id
        conv_u = uuid.UUID(str(conversation_id)) if not isinstance(conversation_id, uuid.UUID) else conversation_id
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conv_u,
                Conversation.organization_id == org_u,
            )
            .options(selectinload(Conversation.messages))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
