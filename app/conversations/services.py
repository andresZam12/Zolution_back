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
from app.core.database import bind_tenant, resolve_whatsapp_tenant
from app.integrations.google.calendar import GoogleCalendarService
from app.integrations.whatsapp.client import WhatsAppClient
from app.llm_providers.base import ConversationMessage
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)

CALENDAR_FAILED_REPLY = (
    "Recibí tu solicitud, pero no pude actualizar la agenda en este momento, así que "
    "este cambio aún no está confirmado. Por favor intenta de nuevo en unos minutos."
)
NEED_DATETIME_REPLY = (
    "Para continuar necesito la fecha y la hora exactas de tu cita. ¿Me las confirmas?"
)
NO_APPOINTMENT_REPLY = (
    "No encontré una cita próxima registrada con este número. ¿Deseas agendar una nueva?"
)

_TIME_PATTERN = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$", re.IGNORECASE)


def _parse_appointment_start(date_arg: str, time_arg: str, now: datetime) -> datetime | None:
    """Parse the date and time of a command; None unless they form a future slot."""
    time_match = _TIME_PATTERN.match(time_arg.strip())
    if not time_match:
        return None

    hour = int(time_match.group(1))
    minute = int(time_match.group(2) or 0)
    meridiem = (time_match.group(3) or "").lower().replace(".", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    try:
        day = datetime.strptime(date_arg.strip(), "%Y-%m-%d")
        start = day.replace(hour=hour, minute=minute)
    except ValueError:
        return None
    return start if start > now else None


def _to_bogota_iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:00-05:00")


def _event_starts_at(event: dict, start_iso: str) -> bool:
    event_start = event.get("start", {}).get("dateTime")
    if not event_start:
        return False
    return datetime.fromisoformat(event_start) == datetime.fromisoformat(start_iso)


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

        # 2. A conversation belongs to one tenant: never reuse another organization's thread
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

        # 1. Route strictly by the receiving number. Unknown numbers are never assigned to a tenant.
        routed_organization_id = await resolve_whatsapp_tenant(self.db, clean_phone_id)
        if not routed_organization_id:
            logger.warning(f"No AgentConfig linked to WhatsApp phone_number_id: {clean_phone_id}")
            return

        await bind_tenant(self.db, routed_organization_id)
        agent_res = await self.db.execute(
            select(AgentConfig).where(AgentConfig.organization_id == routed_organization_id)
        )
        agent_config = agent_res.scalar_one()

        org_res = await self.db.execute(
            select(Organization).where(Organization.id == agent_config.organization_id)
        )
        org = org_res.scalar_one_or_none()

        if agent_config.status != "active" or not org or org.status != "active":
            logger.info(f"Ignored WhatsApp message for inactive agent {agent_config.id}")
            return

        organization_id = str(agent_config.organization_id)

        org_name = org.name or "tu negocio"

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
            .where(
                Message.conversation_id == conversation.id,
                Message.organization_id == agent_config.organization_id,
            )
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
        
        # Build catalog of services and pricing from onboarding answers
        raw_services = ans.get("services")
        services_lines = []
        if isinstance(raw_services, list):
            for s in raw_services:
                if isinstance(s, dict):
                    sn = s.get("name") or s.get("service") or ""
                    sp = s.get("price_cop") or s.get("price") or ""
                    sd = s.get("duration_minutes") or s.get("duration") or ""
                    l = f"- {sn}"
                    if sp:
                        l += f": ${sp:,} COP" if isinstance(sp, (int, float)) else f": {sp}"
                    if sd:
                        l += f" ({sd} min)"
                    services_lines.append(l)
                elif isinstance(s, str) and s.strip():
                    services_lines.append(f"- {s.strip()}")
        elif isinstance(raw_services, str) and raw_services.strip():
            services_lines.append(raw_services.strip())
        services_text = "\n".join(services_lines) if services_lines else "- Asesoría y diseño web / desarrollo de software"

        schedule_text = ans.get("schedule") or ans.get("hours") or "Lunes a Viernes 8:00 AM - 6:00 PM"

        now_co = datetime.utcnow() - timedelta(hours=5)
        tomorrow = (now_co + timedelta(days=1)).strftime("%Y-%m-%d")
        default_avail = f"Atendemos {schedule_text}. Horarios recomendados: 10:00 AM y 3:30 PM."
        avail_info = default_avail

        cal_service = GoogleCalendarService(self.db, organization_id)
        
        # Check if customer already has a scheduled appointment in Google Calendar
        existing_appointment_info = "No hay citas previas agendadas registradas."
        try:
            evs = await cal_service.find_customer_events(customer_phone)
            if evs:
                active_evs = []
                for ev in evs:
                    st = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date") or ""
                    sm = ev.get("summary") or "Cita"
                    if st:
                        active_evs.append(f"{sm} el {st}")
                if active_evs:
                    existing_appointment_info = "; ".join(active_evs[:3])
        except Exception as e_ev:
            logger.info(f"Could not load customer existing appointments: {e_ev}")

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
            f"Catálogo de Servicios y Tarifas:\n{services_text}\n"
            f"Horario de atención: {schedule_text}\n"
            f"Cita(s) agendada(s) del cliente en el calendario: {existing_appointment_info}\n"
            f"Disponibilidad general para agendar ({tomorrow}): {avail_info}\n\n"
            "DIRECTRICES DE RESPUESTA EN WHATSAPP:\n"
            "1. COMUNICACIÓN NATURAL Y COMPLETA: Responde siempre con calidez, empatía y sentido completo. Si el cliente hace varias preguntas en un solo mensaje (ej: pregunta cuándo es su cita Y cuánto cuesta un servicio), responde AMBAS preguntas con claridad y amabilidad.\n"
            "2. LONGITUD: Mantén respuestas ágiles de WhatsApp (de 2 a 4 oraciones bien redactadas), evitando párrafos innecesariamente largos pero NUNCA cortando oraciones ni dejando ideas a medias.\n"
            "3. CONSULTAS DE CITAS EXISTENTES: Si el cliente pregunta cuándo es su cita o pide recordarle el horario, respóndele con la información de su cita registrada en el calendario SIN emitir ningún comando ni tag.\n"
            "4. COMANDOS DE ACCIÓN (SOLO cuando el cliente solicita o confirma una fecha/hora nueva):\n"
            "   - Agendar nueva cita: Confirma la fecha y hora en el texto y al final pon en renglón nuevo: [[AGENDAR|YYYY-MM-DD|HH:MM]] usando la fecha real que pidió el cliente\n"
            "   - Reagendar / cambiar hora: Confirma el nuevo horario en el texto y al final pon en renglón nuevo: [[REAGENDAR|YYYY-MM-DD|HH:MM]]\n"
            "   - Cancelar: Confirma la cancelación en el texto y al final pon en renglón nuevo: [[CANCELAR]]\n"
            "   - Preguntas generales, consultas de precios o agradecimientos: Responde cordialmente SIN agregar ningún comando ni corchetes.\n"
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
                        max_tokens=800,
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

        calendar_reply, booked_start = await self._execute_calendar_command(
            cal_service,
            command_type,
            cmd_args,
            effective_customer_name,
            customer_phone,
            now_co,
        )

        # Clean tags safely from reply_text:
        # Strip ONLY the bracketed command tags, NEVER strip surrounding conversation text
        reply_text = re.sub(r"\[\[\s*[A-Z_]+[^\]]*\]\]?", "", reply_text)
        reply_text = re.sub(r"\[\[.*?\]\]?", "", reply_text)
        # If there's an unclosed bracket tag at the very end of the string, remove only the tail
        reply_text = re.sub(r"\[\[[A-Za-z0-9_|:\- ]*$", "", reply_text)
        # Clean double spaces or excess blank lines
        reply_text = re.sub(r"[ \t]{2,}", " ", reply_text)
        reply_text = re.sub(r"\n{3,}", "\n\n", reply_text).strip()

        # Fix truncated sentences ending in incomplete prepositions or punctuation
        reply_text = re.sub(r"[\s,;:-]+$", "", reply_text).strip()
        incomplete_pattern = r"\b(para mañana|mañana|a las|a|el día|para|las|el|en|confirmada para|quedó agendada para|quedó para)$"
        if booked_start and re.search(incomplete_pattern, reply_text, re.IGNORECASE):
            hh, mm = booked_start.hour, booked_start.minute
            hh_12 = ((hh - 1) % 12) + 1
            ampm = "PM" if hh >= 12 else "AM"
            if re.search(r"\b(a las|a)$", reply_text, re.IGNORECASE):
                reply_text = re.sub(r"\b(a las|a)$", f"a las {hh_12}:{mm:02d} {ampm}.", reply_text)
            else:
                reply_text = f"{reply_text} a las {hh_12}:{mm:02d} {ampm}."
        elif reply_text and not reply_text.endswith((".", "!", "?")):
            reply_text = reply_text + "."

        # The customer is only told about what actually happened on the calendar
        if calendar_reply:
            reply_text = calendar_reply

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

    async def _execute_calendar_command(
        self,
        cal_service: GoogleCalendarService,
        command_type: str | None,
        cmd_args: list[str],
        customer_name: str,
        customer_phone: str,
        now: datetime,
    ) -> tuple[str | None, datetime | None]:
        """
        Run a calendar command before anything is promised to the customer.

        Returns (reply override, booked start). The override is None when the
        model's reply is accurate, otherwise it replaces that reply.
        """
        if not command_type:
            return None, None

        start = None
        if command_type in ("AGENDAR", "REAGENDAR"):
            start = _parse_appointment_start(
                cmd_args[0] if cmd_args else "",
                cmd_args[1] if len(cmd_args) > 1 else "",
                now,
            )
            if not start:
                logger.info(f"Calendar command {command_type} without a valid date/time: {cmd_args}")
                return NEED_DATETIME_REPLY, None

        try:
            events = await cal_service.find_customer_events(customer_phone)

            if command_type == "CANCELAR":
                if not events:
                    return NO_APPOINTMENT_REPLY, None
                result = await cal_service.delete_appointment(events[0]["id"])
                if not result.get("success"):
                    logger.error(f"Could not cancel event {events[0]['id']}: {result}")
                    return CALENDAR_FAILED_REPLY, None
                return None, None

            if command_type == "REAGENDAR" and not events:
                return NO_APPOINTMENT_REPLY, None

            start_iso = _to_bogota_iso(start)
            if any(_event_starts_at(ev, start_iso) for ev in events):
                logger.info(f"Customer already has an event at {start_iso}; nothing to create")
                return None, start

            created = await cal_service.create_appointment(
                customer_name=customer_name,
                customer_phone=customer_phone,
                start_time=start_iso,
                end_time=_to_bogota_iso(start + timedelta(hours=1)),
                notes=cmd_args[3] if len(cmd_args) > 3 else "Asesoría especializada",
            )
            if not created.get("success"):
                logger.error(f"Could not create calendar event: {created.get('message')}")
                return CALENDAR_FAILED_REPLY, None

            # The previous slot is released only once the new one exists
            if command_type == "REAGENDAR":
                removed = await cal_service.delete_appointment(events[0]["id"])
                if not removed.get("success"):
                    logger.error(f"Rescheduled but old event remains {events[0]['id']}: {removed}")

            return None, start
        except Exception as exc:
            logger.error(f"Calendar command {command_type} failed: {exc}")
            return CALENDAR_FAILED_REPLY, None

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
