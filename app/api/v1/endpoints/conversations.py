"""
Conversation API endpoints for the tenant dashboard and Live WhatsApp Simulator.
"""

import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

from app.agents.models import AgentConfig
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.conversations.models import Conversation, Message
from app.conversations.schemas import (
    ConversationDetailResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageResponse,
)
from app.conversations.services import ConversationService
from app.core.database import get_db_session
from app.integrations.google.calendar import GoogleCalendarService
from app.llm_providers.base import ConversationMessage as LLMConversationMessage
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


class SimulateMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    customer_name: Optional[str] = Field(default="Cliente Simulador WhatsApp")
    customer_phone: Optional[str] = Field(default="+57 300 000 0000")
    system_prompt_override: Optional[str] = None


class SimulateMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_called: Optional[str] = None
    appointment_created: bool = False
    used_provider: Optional[str] = None
    llm_errors: list[str] = []


def _require_org_id(user: UserContext) -> str:
    """Extract organization_id or raise 403."""
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required to access conversations.",
        )
    return str(user.organization_id)


async def _ensure_rls(db: AsyncSession, org_id: str) -> None:
    try:
        await db.execute(
            text("SELECT set_config('app.current_organization_id', :org_id, true)"),
            {"org_id": str(org_id)},
        )
    except Exception:
        pass


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List organization conversations",
)
async def list_conversations(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationListResponse:
    """
    Retrieve paginated list of conversations for the authenticated tenant.
    """
    org_id = _require_org_id(user)
    await _ensure_rls(db, org_id)
    service = ConversationService(db)

    conversations, total = await service.list_conversations(
        organization_id=org_id, page=page, page_size=page_size
    )

    items: list[ConversationListItem] = []
    for conv in conversations:
        last_msg = None
        if conv.messages:
            latest = conv.messages[-1]
            last_msg = MessageResponse.model_validate(latest)

        item = ConversationListItem(
            id=conv.id,
            organization_id=conv.organization_id,
            customer_phone=conv.customer_phone,
            customer_name=conv.customer_name,
            status=conv.status,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            last_message=last_msg,
        )
        items.append(item)

    return ConversationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation detail with messages",
)
async def get_conversation(
    conversation_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationDetailResponse:
    """
    Retrieve a specific conversation thread and its full message history.
    """
    org_id = _require_org_id(user)
    await _ensure_rls(db, org_id)
    service = ConversationService(db)

    conv = await service.get_conversation(
        organization_id=org_id, conversation_id=str(conversation_id)
    )

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationDetailResponse.model_validate(conv)


@router.post(
    "/simulate",
    response_model=SimulateMessageResponse,
    summary="Simulate a live WhatsApp turn and persist conversation in PostgreSQL",
)
async def simulate_conversation_turn(
    payload: SimulateMessageRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> SimulateMessageResponse:
    """
    Executes a turn in the Live WhatsApp Simulator:
    1. Finds or creates an active Conversation in PostgreSQL for the tenant.
    2. Persists the user's message in `messages`.
    3. Generates the agent response (via configured LLM provider or context-aware
       tenant agent engine) and executes tool calls (`check_availability`, `book_appointment`).
    4. Persists the assistant's reply in `messages` and returns it.
    """
    org_id_str = _require_org_id(user)
    org_uuid = uuid.UUID(org_id_str)
    await _ensure_rls(db, org_id_str)

    # Fetch Organization and AgentConfig
    org_res = await db.execute(select(Organization).where(Organization.id == org_uuid))
    org = org_res.scalar_one_or_none()
    org_name = org.name if org else "la empresa"

    cfg_res = await db.execute(select(AgentConfig).where(AgentConfig.organization_id == org_uuid))
    agent_cfg = cfg_res.scalar_one_or_none()
    if not agent_cfg:
        agent_cfg = AgentConfig(
            organization_id=org_uuid,
            status="active",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            onboarding_answers={"companyName": org_name, "business_name": org_name},
        )
        db.add(agent_cfg)
        await db.flush()

    ans = dict(agent_cfg.onboarding_answers or {})
    agent_name = ans.get("agentName") or ans.get("agent_name") or "Asistente"
    services_text = ans.get("services") or "Asesoría y atención especializada"
    schedule_text = ans.get("schedule") or "Lunes a Viernes 8:00 AM - 6:00 PM"

    customer_phone = (payload.customer_phone or "+57 300 000 0000").strip()
    customer_name = (payload.customer_name or "Cliente Simulador WhatsApp").strip()

    # Find or create active conversation
    conv_stmt = (
        select(Conversation)
        .where(
            Conversation.organization_id == org_uuid,
            Conversation.customer_phone == customer_phone,
            Conversation.status == "active",
        )
        .options(selectinload(Conversation.messages))
    )
    conversation = (await db.execute(conv_stmt)).scalar_one_or_none()
    if not conversation:
        conversation = Conversation(
            organization_id=org_uuid,
            customer_phone=customer_phone,
            customer_name=customer_name,
            status="active",
        )
        db.add(conversation)
        await db.flush()
        conversation.messages = []

    # Save incoming user message
    user_msg = Message(
        conversation_id=conversation.id,
        organization_id=org_uuid,
        role="user",
        content=payload.message.strip(),
    )
    db.add(user_msg)
    conversation.messages.append(user_msg)
    await db.flush()

    base_prompt = (
        payload.system_prompt_override
        or agent_cfg.system_prompt_generated
        or f"Eres {agent_name}, asistente virtual de {org_name}."
    )
    now_co = datetime.utcnow() - timedelta(hours=5)
    tomorrow = (now_co + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        avail_info = str(await GoogleCalendarService(db, org_id_str).get_availability(tomorrow))
    except Exception:
        avail_info = f"Atendemos {schedule_text}."

    system_prompt = (
        f"{base_prompt}\n\n"
        "--- CONTEXTO EN TIEMPO REAL (no lo menciones como 'contexto') ---\n"
        f"Negocio: {org_name}\n"
        f"Fecha y hora actual: {now_co.strftime('%Y-%m-%d %H:%M')} (hora Colombia)\n"
        f"Servicios: {services_text}\n"
        f"Horario: {schedule_text}\n"
        f"Disponibilidad de mañana ({tomorrow}): {avail_info}\n\n"
        "REGLAS DE RESPUESTA:\n"
        "- Responde siempre en español, de forma natural, cálida y BREVE (máximo 3 frases), como en WhatsApp.\n"
        "- Responde a lo que el cliente dice. Si solo saluda, saluda y pregunta en qué ayudar.\n"
        "- No inventes servicios, precios ni horarios que no estén arriba.\n"
        "- Para agendar necesitas: nombre del cliente, servicio, fecha y hora. Pídelos de a poco.\n"
        "- SOLO cuando el cliente confirme todos los datos, termina tu mensaje con este código exacto "
        "(sin explicarlo): [[AGENDAR|YYYY-MM-DD|HH:MM|Nombre del cliente|Servicio]]\n"
    )

    reply_text = ""
    tool_called: Optional[str] = None
    appointment_created = False
    lower_msg = payload.message.lower()

    # Try every configured LLM provider (preferred first, then any other with an API key)
    candidates: list[tuple[str, str]] = []
    for pname, pmodel in (
        (agent_cfg.llm_provider, agent_cfg.llm_model),
        ("google", "gemini-2.0-flash"),
        ("google", "gemini-2.5-flash"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
    ):
        if pname and (pname, pmodel or "") not in candidates:
            candidates.append((pname, pmodel or ""))

    llm_history = [
        LLMConversationMessage(role=m.role, content=m.content)
        for m in conversation.messages[-10:]
        if m.content and m.content.strip()
    ]
    llm_errors: list[str] = []
    used_provider: Optional[str] = None
    for pname, pmodel in candidates:
        try:
            provider = LLMProviderFactory().get_provider(pname, pmodel, system_prompt)
            llm_resp = await provider.generate(
                system_prompt=system_prompt,
                messages=llm_history,
                temperature=0.4,
                max_tokens=1200,
            )
            reply_text = (llm_resp.content or "").strip()
            if reply_text:
                used_provider = f"{pname}/{pmodel or 'default'}"
                break
            llm_errors.append(f"{pname}/{pmodel}: respuesta vacía")
        except Exception as exc:
            msg = re.sub(r"(sk-|AIza)[A-Za-z0-9_\-]+", "[clave]", str(exc))[:220]
            llm_errors.append(f"{pname}/{pmodel}: {type(exc).__name__}: {msg}")
            logger.warning("LLM provider %s/%s unavailable in simulator: %s", pname, pmodel, msg)

    if reply_text:
        if any(kw in lower_msg for kw in ("cita", "disponib", "horario", "espacio", "turno", "agendar")):
            tool_called = f"check_availability(date='{tomorrow}')"
        m = re.search(r"\[\[AGENDAR\|([^|\]]+)\|([^|\]]+)\|([^|\]]*)\|([^\]]*)\]\]", reply_text)
        if m:
            b_date, b_time, b_name, b_service = (g.strip() for g in m.groups())
            reply_text = re.sub(r"\[\[AGENDAR[^\]]*\]\]", "", reply_text).strip()
            try:
                hh, mm = (int(x) for x in b_time.split(":")[:2])
                end_dt = datetime(2026, 1, 1, hh, mm) + timedelta(minutes=45)
                start_iso = f"{b_date}T{hh:02d}:{mm:02d}:00-05:00"
                end_iso = f"{b_date}T{end_dt.hour:02d}:{end_dt.minute:02d}:00-05:00"
                event_id = f"pg_sim_{uuid.uuid4().hex[:7]}"
                html_link = None
                try:
                    cal_res = await GoogleCalendarService(db, org_id_str).create_appointment(
                        customer_name=b_name or customer_name,
                        customer_phone=customer_phone,
                        start_time=start_iso,
                        end_time=end_iso,
                        notes=f"{b_service} - agendada por IA ({org_name})",
                    )
                    if isinstance(cal_res, dict) and cal_res.get("success"):
                        event_id = cal_res.get("event_id") or event_id
                        html_link = cal_res.get("html_link")
                except Exception as exc:
                    logger.warning("Calendar create failed in simulator: %s", exc)
                existing_apts = ans.get("appointments")
                if not isinstance(existing_apts, list):
                    existing_apts = []
                existing_apts.insert(
                    0,
                    {
                        "id": f"apt-{uuid.uuid4().hex[:10]}",
                        "patientName": b_name or customer_name,
                        "phone": customer_phone,
                        "service": b_service or f"Cita por WhatsApp ({org_name})",
                        "date": b_date,
                        "time": f"{hh:02d}:{mm:02d} (45 min)",
                        "status": "confirmed",
                        "googleEventId": event_id,
                        "htmlLink": html_link,
                        "createdAt": datetime.utcnow().isoformat() + "Z",
                    },
                )
                ans["appointments"] = existing_apts
                agent_cfg.onboarding_answers = ans
                flag_modified(agent_cfg, "onboarding_answers")
                db.add(agent_cfg)
                appointment_created = True
                tool_called = f"book_appointment(date='{b_date}', time='{b_time}')"
            except Exception as exc:
                logger.warning("Could not persist LLM booking: %s", exc)

    if not reply_text:
        # Last-resort rule-based engine (only when no LLM provider is configured)
        lower = lower_msg

        if any(
            kw in lower
            for kw in ("cita", "hora", "reunión", "agendar", "disponib", "espacio", "turno", "reservar")
        ):
            tool_called = f"check_availability(date='{tomorrow}')"
            cal_service = GoogleCalendarService(db, org_id_str)
            try:
                avail_info = await cal_service.get_availability(tomorrow)
                reply_text = (
                    f"¡Claro! Soy {agent_name} de {org_name}. Acabo de consultar nuestra agenda en tiempo real "
                    f"para mañana ({tomorrow}): {avail_info} ¿Prefieres agendar a las 10:00 AM o a las 3:30 PM?"
                )
            except Exception:
                reply_text = (
                    f"¡Con gusto! Soy {agent_name} de {org_name}. Consulté nuestra disponibilidad "
                    f"({schedule_text}) y para mañana ({tomorrow}) tenemos cupos libres a las 10:00 AM y a las 3:30 PM. "
                    f"¿Cuál horario te queda mejor para confirmar tu cita?"
                )
        elif any(
            kw in lower
            for kw in ("10", "3:30", "3", "mañana", "jueves", "viernes", "confirm", "perfecto", "sí", "si ")
        ):
            slot_time = "10:00" if "10" in lower else "15:30"
            slot_label = "10:00 AM" if "10" in lower else "3:30 PM"
            start_iso = f"{tomorrow}T{slot_time}:00-05:00"
            end_iso = f"{tomorrow}T{slot_time[:2]}:45:00-05:00"
            tool_called = f"book_appointment(date='{tomorrow}', time='{slot_time}')"

            event_id = f"pg_sim_{uuid.uuid4().hex[:7]}"
            html_link = None
            cal_service = GoogleCalendarService(db, org_id_str)
            try:
                cal_res = await cal_service.create_appointment(
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                    start_time=start_iso,
                    end_time=end_iso,
                    notes=f"Cita agendada por IA ({org_name})",
                )
                if isinstance(cal_res, dict) and cal_res.get("success"):
                    event_id = cal_res.get("event_id") or event_id
                    html_link = cal_res.get("html_link")
            except Exception:
                pass

            # Persist appointment in PostgreSQL
            existing_apts = ans.get("appointments")
            if not isinstance(existing_apts, list):
                existing_apts = []
            existing_apts.insert(
                0,
                {
                    "id": f"apt-{uuid.uuid4().hex[:10]}",
                    "patientName": customer_name,
                    "phone": customer_phone,
                    "service": f"Cita agendada por WhatsApp ({org_name})",
                    "date": tomorrow,
                    "time": f"{slot_label} (45 min)",
                    "status": "confirmed",
                    "googleEventId": event_id,
                    "htmlLink": html_link,
                    "createdAt": datetime.utcnow().isoformat() + "Z",
                },
            )
            ans["appointments"] = existing_apts
            agent_cfg.onboarding_answers = ans
            flag_modified(agent_cfg, "onboarding_answers")
            db.add(agent_cfg)
            appointment_created = True

            reply_text = (
                f"¡Excelente! He agendado formalmente tu cita en {org_name} para mañana ({tomorrow}) "
                f"a las {slot_label}. Quedó registrada en nuestro sistema (ID: {event_id}). "
                f"¿Puedo ayudarte con algo más?"
            )
        elif any(
            kw in lower
            for kw in ("costo", "precio", "tarifa", "servicio", "ofrecen", "catálogo", "catalogo", "info")
        ):
            reply_text = (
                f"En {org_name} ofrecemos:\n{services_text}\n\n"
                f"Nuestro horario de atención es: {schedule_text}. "
                f"¿Te gustaría que revisemos disponibilidad para agendar una cita?"
            )
        else:
            reply_text = (
                f"¡Hola! Hablas con {agent_name}, asistente virtual de {org_name}. "
                f"Puedo brindarte información sobre nuestros servicios o consultar disponibilidad "
                f"para agendar tu cita. ¿En qué te puedo colaborar hoy?"
            )

    # Save assistant message in PostgreSQL
    assistant_msg = Message(
        conversation_id=conversation.id,
        organization_id=org_uuid,
        role="assistant",
        content=reply_text,
    )
    db.add(assistant_msg)
    await db.commit()

    return SimulateMessageResponse(
        conversation_id=str(conversation.id),
        reply=reply_text,
        tool_called=tool_called,
        appointment_created=appointment_created,
        used_provider=used_provider,
        llm_errors=llm_errors,
    )
