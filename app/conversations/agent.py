"""
The agent's single brain.

WhatsApp and the simulator run the same turn here, so testing in the simulator
tells the owner how the agent will behave with real customers. Only the callers
differ: how the customer message arrives and where the reply goes.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.appointments.services import (
    BOGOTA,
    AppointmentService,
    CommandResult,
    parse_command,
    strip_command_tags,
)
from app.conversations.models import Message
from app.core.time_format import format_time_12h
from app.integrations.google.calendar import GoogleCalendarService
from app.llm_providers.base import ConversationMessage
from app.llm_providers.factory import LLMProviderFactory

logger = logging.getLogger(__name__)

LLM_CANDIDATES = [
    ("google", "gemini-flash-latest"),
    ("openai", "gpt-4o-mini"),
]
LLM_TIMEOUT_SECONDS = 12.0
# Gemini's thinking counts against this limit. Brevity comes from the prompt instructions.
LLM_MAX_TOKENS = 2048
AVAILABILITY_TIMEOUT_SECONDS = 2.0
HISTORY_LENGTH = 12

MODEL_UNAVAILABLE_REPLY = (
    "Gracias por escribirnos. En este momento no puedo procesar tu mensaje. "
    "Por favor intenta de nuevo en unos minutos."
)
NO_CATALOG_TEXT = (
    "- Catálogo no configurado: no inventes servicios ni precios, "
    "y ofrece que el negocio confirme esa información."
)

_CALENDAR_KEYWORDS = (
    "cita",
    "hora",
    "agendar",
    "disponib",
    "espacio",
    "turno",
    "reservar",
    "mañana",
    "calendario",
    "cupo",
)
_INCOMPLETE_ENDING = (
    r"\b(para mañana|mañana|a las|a|el día|para|las|el|en|confirmada para|"
    r"quedó agendada para|quedó para)$"
)


@dataclass
class TurnResult:
    reply: str
    outcome: CommandResult
    command_type: str | None = None
    model_answered: bool = False
    used_provider: str | None = None
    errors: list[str] = field(default_factory=list)


def local_now() -> datetime:
    """Current Colombian time as a naive datetime, as calendar commands expect it."""
    return datetime.now(BOGOTA).replace(tzinfo=None)


def _services_text(answers: dict) -> str:
    raw_services = answers.get("services")
    lines = []
    if isinstance(raw_services, list):
        for service in raw_services:
            if isinstance(service, dict):
                name = service.get("name") or service.get("service") or ""
                price = service.get("price_cop") or service.get("price") or ""
                duration = service.get("duration_minutes") or service.get("duration") or ""
                line = f"- {name}"
                if price:
                    line += f": ${price:,} COP" if isinstance(price, int | float) else f": {price}"
                if duration:
                    line += f" ({duration} min)"
                lines.append(line)
            elif isinstance(service, str) and service.strip():
                lines.append(f"- {service.strip()}")
    elif isinstance(raw_services, str) and raw_services.strip():
        lines.append(raw_services.strip())
    return "\n".join(lines) if lines else NO_CATALOG_TEXT


def _build_system_prompt(
    base_prompt: str,
    org_name: str,
    customer_name: str,
    customer_phone: str,
    answers: dict,
    appointments_text: str,
    availability_date: str,
    availability_text: str,
    now: datetime,
) -> str:
    schedule = (
        answers.get("schedule") or answers.get("hours") or "Lunes a Viernes 8:00 AM - 6:00 PM"
    )
    current_time = format_time_12h(now.replace(tzinfo=BOGOTA))
    return (
        f"{base_prompt}\n\n"
        "--- CONTEXTO EN TIEMPO REAL ---\n"
        f"Negocio: {org_name}\n"
        f"Fecha y hora actual: {now:%Y-%m-%d} {current_time} (hora Colombia)\n"
        f"Cliente con quien hablas: {customer_name}\n"
        f"Teléfono del cliente: {customer_phone}\n"
        f"Catálogo de Servicios y Tarifas:\n{_services_text(answers)}\n"
        f"Horario de atención: {schedule}\n"
        f"Cita(s) agendada(s) del cliente en el calendario: {appointments_text}\n"
        f"Disponibilidad general para agendar ({availability_date}): {availability_text}\n\n"
        "DIRECTRICES DE RESPUESTA EN WHATSAPP:\n"
        "FORMATO DE HORA: al escribirle al cliente usa siempre el formato de 12 horas (ej: 3:00 PM), "
        "nunca el de 24 horas. El formato HH:MM de 24 horas se usa solamente dentro de los comandos "
        "entre corchetes.\n"
        "1. COMUNICACIÓN NATURAL Y COMPLETA: Responde siempre con calidez, empatía y sentido completo. "
        "Si el cliente hace varias preguntas en un solo mensaje (ej: pregunta cuándo es su cita Y "
        "cuánto cuesta un servicio), responde AMBAS preguntas con claridad y amabilidad.\n"
        "2. LONGITUD: Mantén respuestas ágiles de WhatsApp (de 2 a 4 oraciones bien redactadas), "
        "evitando párrafos innecesariamente largos pero NUNCA cortando oraciones ni dejando ideas a "
        "medias.\n"
        "3. CONSULTAS DE CITAS EXISTENTES: Si el cliente pregunta cuándo es su cita o pide recordarle "
        "el horario, respóndele con la información de su cita registrada en el calendario SIN emitir "
        "ningún comando ni tag.\n"
        "4. COMANDOS DE ACCIÓN (SOLO cuando el cliente solicita o confirma una fecha/hora nueva):\n"
        "   - Agendar nueva cita: Confirma la fecha y hora en el texto y al final pon en renglón "
        "nuevo: [[AGENDAR|YYYY-MM-DD|HH:MM]] usando la fecha real que pidió el cliente\n"
        "   - Reagendar / cambiar hora: Confirma el nuevo horario en el texto y al final pon en "
        "renglón nuevo: [[REAGENDAR|YYYY-MM-DD|HH:MM]]\n"
        "   - Cancelar: Confirma la cancelación en el texto y al final pon en renglón nuevo: [[CANCELAR]]\n"
        "   - Preguntas generales, consultas de precios o agradecimientos: Responde cordialmente SIN "
        "agregar ningún comando ni corchetes.\n"
    )


async def _load_history(
    db: AsyncSession, conversation_id: uuid.UUID, organization_id: uuid.UUID
) -> list[ConversationMessage]:
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.organization_id == organization_id,
        )
        .order_by(Message.created_at.desc())
        .limit(HISTORY_LENGTH)
    )
    history = []
    for message in reversed(result.scalars().all()):
        content = (message.content or "").strip()
        if not content:
            continue
        if message.role == "assistant":
            content = re.sub(r"\[\[.*?\]\]", "", content).strip()
            content = re.sub(r"[\s,;:-]+$", "", content).strip()
            if not content.endswith((".", "!", "?")):
                content += "."
        history.append(ConversationMessage(role=message.role, content=content))
    return history


async def _availability(
    calendar: GoogleCalendarService, text_content: str, date: str, default: str
) -> str:
    lowered = text_content.lower()
    if not any(keyword in lowered for keyword in _CALENDAR_KEYWORDS):
        return default
    try:
        availability = await asyncio.wait_for(
            calendar.get_availability(date), timeout=AVAILABILITY_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.info(f"Calendar availability check skipped/failed: {exc}")
        return default
    return str(availability) if "no disponible" not in str(availability).lower() else default


def _last_complete_sentences(text: str) -> str:
    """Keep what a cut-off reply fully said. An unfinished command makes it unusable."""
    if text.count("[[") != text.count("]]"):
        return ""
    last_end = max(text.rfind(mark) for mark in ".!?")
    return text[: last_end + 1].strip() if last_end >= 0 else ""


async def _ask_model(
    factory: LLMProviderFactory, system_prompt: str, history: list[ConversationMessage]
) -> tuple[str, str | None, list[str]]:
    errors = []
    partial, partial_provider = "", None
    for provider_name, model in LLM_CANDIDATES:
        label = f"{provider_name}/{model}"
        try:
            provider = factory.get_provider(provider_name, model, system_prompt)
            response = await asyncio.wait_for(
                provider.generate(
                    system_prompt=system_prompt,
                    messages=history,
                    temperature=0.3,
                    max_tokens=LLM_MAX_TOKENS,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = (response.content or "").strip()
            if content and not response.truncated:
                return content, label, errors
            if content and not partial:
                partial, partial_provider = content, label
            errors.append(f"{label}: {'respuesta cortada' if content else 'respuesta vacía'}")
        except Exception as exc:
            safe_message = re.sub(r"(sk-|AIza)[A-Za-z0-9_\-]+", "[clave]", str(exc))[:220]
            errors.append(f"{label}: {type(exc).__name__}: {safe_message}")
            logger.warning(f"LLM candidate {label} failed: {safe_message}")

    usable = _last_complete_sentences(partial)
    return usable, partial_provider if usable else None, errors


def _complete_sentence(reply: str, booked_start: datetime | None) -> str:
    """Close replies the model cut short, using the real booked time when there is one."""
    reply = re.sub(r"[\s,;:-]+$", "", reply).strip()
    if booked_start and re.search(_INCOMPLETE_ENDING, reply, re.IGNORECASE):
        booked_time = format_time_12h(booked_start)
        if re.search(r"\b(a las|a)$", reply, re.IGNORECASE):
            return re.sub(r"\b(a las|a)$", f"a las {booked_time}.", reply)
        return f"{reply} a las {booked_time}."
    if reply and reply[-1].isalnum():
        return f"{reply}."
    return reply


async def run_agent_turn(
    db: AsyncSession,
    *,
    agent_config: AgentConfig,
    org_name: str,
    conversation_id: uuid.UUID,
    customer_name: str,
    customer_phone: str,
    text_content: str,
    source: str,
    calendar: GoogleCalendarService,
    llm_factory: LLMProviderFactory,
    require_calendar: bool,
    prompt_override: str | None = None,
) -> TurnResult:
    """Answer one customer message. The message must already be saved in the conversation."""
    answers = dict(agent_config.onboarding_answers or {})
    agent_name = answers.get("agentName") or answers.get("agent_name") or "Asistente"
    now = local_now()
    availability_date = f"{now + timedelta(days=1):%Y-%m-%d}"
    schedule = (
        answers.get("schedule") or answers.get("hours") or "Lunes a Viernes 8:00 AM - 6:00 PM"
    )

    appointments = AppointmentService(db, agent_config.organization_id, calendar)
    upcoming = await appointments.upcoming_for_customer(customer_phone)
    appointments_text = (
        "; ".join(
            f"{a.service} el {a.starts_at.astimezone(BOGOTA):%Y-%m-%d} a las {format_time_12h(a.starts_at)}"
            for a in upcoming[:3]
        )
        or "No hay citas previas agendadas registradas."
    )

    history = await _load_history(db, conversation_id, agent_config.organization_id)
    availability_text = await _availability(
        calendar,
        text_content,
        availability_date,
        f"Atendemos {schedule}. Horarios recomendados: 10:00 AM y 3:30 PM.",
    )
    system_prompt = _build_system_prompt(
        prompt_override
        or agent_config.system_prompt_generated
        or f"Eres {agent_name}, asistente virtual de {org_name}.",
        org_name,
        customer_name,
        customer_phone,
        answers,
        appointments_text,
        availability_date,
        availability_text,
        now,
    )

    # The model can take seconds: don't hold a database connection meanwhile
    await db.commit()
    raw_reply, used_provider, errors = await _ask_model(llm_factory, system_prompt, history)

    command_type, cmd_args = parse_command(raw_reply)
    outcome = await appointments.execute_command(
        command_type,
        cmd_args,
        customer_name,
        customer_phone,
        now,
        source,
        require_calendar=require_calendar,
    )

    reply = _complete_sentence(strip_command_tags(raw_reply), outcome.booked_start)
    return TurnResult(
        reply=outcome.reply_override or reply or MODEL_UNAVAILABLE_REPLY,
        outcome=outcome,
        command_type=command_type,
        model_answered=bool(raw_reply),
        used_provider=used_provider,
        errors=errors,
    )
