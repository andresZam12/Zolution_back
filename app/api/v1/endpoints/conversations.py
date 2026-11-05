"""
Conversation API endpoints for the tenant dashboard and Live WhatsApp Simulator.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.models import AgentConfig
from app.appointments.services import AppointmentService, parse_command, strip_command_tags
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
from app.core.database import bind_tenant, get_db_session
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
    session_id: Optional[str] = None
    reset_session: Optional[bool] = False
    system_prompt_override: Optional[str] = None
    llm_provider_override: Optional[str] = None
    llm_model_override: Optional[str] = None


class SimulateMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_called: Optional[str] = None
    appointment_created: bool = False
    google_calendar_synced: bool = False
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
    await bind_tenant(db, org_id)
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
    await bind_tenant(db, org_id)
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
    Guaranteed never to crash with 500 — graceful fallbacks at every stage.
    """
    fallback_conv_id = str(uuid.uuid4())
    org_name = "tu negocio"

    try:
        org_id_str = _require_org_id(user)
        org_uuid = uuid.UUID(org_id_str)
        await bind_tenant(db, org_id_str)

        # 1. Fetch or create Organization
        org = None
        try:
            org_res = await db.execute(select(Organization).where(Organization.id == org_uuid))
            org = org_res.scalar_one_or_none()
            if not org:
                org = Organization(
                    id=org_uuid,
                    name="Mi Negocio",
                    slug=f"org-{org_uuid.hex[:8]}",
                    plan="starter",
                    status="active",
                )
                db.add(org)
                await db.flush()
            org_name = org.name or "tu negocio"
        except Exception as org_exc:
            logger.warning("Could not ensure Organization row: %s", org_exc)
            org_name = "tu negocio"

        # 2. Fetch or create AgentConfig
        agent_cfg = None
        try:
            cfg_res = await db.execute(select(AgentConfig).where(AgentConfig.organization_id == org_uuid))
            agent_cfg = cfg_res.scalar_one_or_none()
            if not agent_cfg:
                agent_cfg = AgentConfig(
                    id=uuid.uuid4(),
                    organization_id=org_uuid,
                    status="active",
                    llm_provider="openai",
                    llm_model="gpt-4o-mini",
                    onboarding_answers={"companyName": org_name, "business_name": org_name},
                )
                db.add(agent_cfg)
                await db.flush()
        except Exception as cfg_exc:
            logger.warning("Could not ensure AgentConfig: %s", cfg_exc)

        ans = dict((agent_cfg.onboarding_answers if agent_cfg else {}) or {})
        agent_name = ans.get("agentName") or ans.get("agent_name") or "Asistente"
        services_text = ans.get("services") or "Asesoría y atención especializada"
        schedule_text = ans.get("schedule") or "Lunes a Viernes 8:00 AM - 6:00 PM"

        session_id = (payload.session_id or "").strip()
        effective_phone = f"+57-sim-{session_id}" if session_id else (payload.customer_phone or "+57 300 000 0000").strip()
        customer_phone = effective_phone
        customer_name = (payload.customer_name or "Cliente Simulador WhatsApp").strip()

        # If reset is requested, archive old active conversation so this session starts completely clean
        if payload.reset_session:
            try:
                from sqlalchemy import update as sql_update
                await db.execute(
                    sql_update(Conversation)
                    .where(
                        Conversation.organization_id == org_uuid,
                        Conversation.customer_phone == effective_phone,
                    )
                    .values(status="archived")
                )
                await db.flush()
            except Exception:
                pass

        # 3. Find or create active conversation
        conversation = None
        if not payload.reset_session:
            try:
                conv_stmt = (
                    select(Conversation)
                    .where(
                        Conversation.organization_id == org_uuid,
                        Conversation.customer_phone == customer_phone,
                        Conversation.status == "active",
                    )
                    .order_by(Conversation.updated_at.desc())
                )
                conv_res = await db.execute(conv_stmt)
                conversation = conv_res.scalars().first()
            except Exception as conv_exc:
                logger.warning("Could not ensure Conversation in DB: %s", conv_exc)

        if not conversation:
            conversation = Conversation(
                id=uuid.uuid4(),
                organization_id=org_uuid,
                customer_phone=customer_phone,
                customer_name=customer_name,
                status="active",
            )
            db.add(conversation)
            await db.flush()

        active_conv_id = conversation.id if conversation else uuid.uuid4()

        # 4. Save incoming user message
        try:
            user_msg = Message(
                id=uuid.uuid4(),
                conversation_id=active_conv_id,
                organization_id=org_uuid,
                role="user",
                content=payload.message.strip(),
            )
            db.add(user_msg)
            await db.flush()
        except Exception as msg_exc:
            logger.warning("Could not persist incoming user message: %s", msg_exc)

        # 5. Fetch message history for LLM
        history_msgs: list[LLMConversationMessage] = []
        try:
            hist_stmt = (
                select(Message)
                .where(Message.conversation_id == active_conv_id)
                .order_by(Message.created_at.asc())
            )
            msg_rows = (await db.execute(hist_stmt)).scalars().all()
            for m in msg_rows[-8:]:
                if m.content and m.content.strip():
                    history_msgs.append(LLMConversationMessage(role=m.role, content=m.content.strip()))
        except Exception as hist_exc:
            logger.warning("Could not read message history: %s", hist_exc)

        if not any(m.role == "user" for m in history_msgs):
            history_msgs.append(LLMConversationMessage(role="user", content=payload.message.strip()))

        # 6. Compose system prompt
        base_prompt = (
            payload.system_prompt_override
            or (agent_cfg.system_prompt_generated if agent_cfg else None)
            or f"Eres {agent_name}, asistente virtual de {org_name}."
        )
        now_co = datetime.utcnow() - timedelta(hours=5)
        tomorrow = (now_co + timedelta(days=1)).strftime("%Y-%m-%d")
        default_avail = f"Atendemos {schedule_text}. Horarios recomendados para ofrecer al cliente: 10:00 AM y 3:30 PM."
        avail_info = default_avail
        lower_msg = payload.message.lower()

        needs_calendar = any(
            kw in lower_msg
            for kw in ("cita", "hora", "agendar", "disponib", "espacio", "turno", "reservar", "mañana", "calendario", "cupo")
        )
        if needs_calendar:
            try:
                avail_res = await asyncio.wait_for(
                    GoogleCalendarService(db, org_id_str).get_availability(tomorrow),
                    timeout=1.5,
                )
                res_str = str(avail_res)
                if res_str and "no disponible" not in res_str.lower() and "no está conectado" not in res_str.lower() and "failed" not in res_str.lower():
                    avail_info = res_str
                else:
                    avail_info = default_avail
            except Exception:
                avail_info = default_avail

        system_prompt = (
            f"{base_prompt}\n\n"
            "--- CONTEXTO EN TIEMPO REAL (no lo menciones como 'contexto') ---\n"
            f"Negocio: {org_name}\n"
            f"Fecha y hora actual: {now_co.strftime('%Y-%m-%d %H:%M')} (hora Colombia)\n"
            f"Cliente con quien hablas: {customer_name}\n"
            f"Servicios: {services_text}\n"
            f"Horario de atención: {schedule_text}\n"
            f"Disponibilidad de citas ({tomorrow}): {avail_info}\n\n"
            "REGLAS OBLIGATORIAS DE RESPUESTA:\n"
            "- Hablas directamente con un cliente por WhatsApp. Sé siempre muy amable, cálido, conversacional y servicial.\n"
            "- Termina siempre tus oraciones e ideas de forma 100% completa y coherente (máximo 2 a 3 frases breves). NUNCA cortes palabras a la mitad.\n"
            "- Si el cliente saluda (ej: 'hola', 'buenas'), salúdalo amablemente, preséntate brevemente y pregúntale en qué le puedes colaborar hoy.\n"
            "- Si el cliente consulta disponibilidad o pide cita o reunión, dale opciones claras de inmediato (ej: 'Tenemos disponibilidad a las 10:00 AM o a las 3:30 PM, ¿cuál te queda mejor?').\n"
            f"- Ya conoces el nombre del cliente ({customer_name}). Al agendar, NO le vuelvas a pedir su nombre ni su correo, confírmale directamente que la cita o reunión virtual ha quedado agendada.\n"
            "- OBLIGATORIO: Cuando el cliente elija o confirme una fecha u hora (ej: 'el jueves a las 11 am', 'a las 10 am', 'agenda por favor'), "
            "confirma de inmediato con entusiasmo que su cita o reunión ha quedado reservada y coloca SIEMPRE al final de tu mensaje el comando: "
            f"[[AGENDAR|YYYY-MM-DD|HH:MM|{customer_name}|Servicio]]\n"
            "- OBLIGATORIO CANCELACIONES: Si el cliente solicita cancelar o anular su cita (ej: 'cancela mi cita', 'ya no puedo ir', 'anula la cita', 'deseo cancelar'), "
            "confirma con amabilidad y calidez que su cita ha sido cancelada y coloca SIEMPRE al final el comando: "
            f"[[CANCELAR|{customer_name}]]\n"
            "- OBLIGATORIO CAMBIO DE FECHA / HORA (REPROGRAMACIÓN): Si el cliente ya tenía una cita y pide cambiar la hora o fecha (ej: 'cámbiame la cita a las 4pm', 'mejor el viernes a las 11', 'mover cita', 'reprogramar'), "
            "confirma con amabilidad el cambio de horario y coloca SIEMPRE al final el comando: "
            f"[[REAGENDAR|YYYY-MM-DD|HH:MM|{customer_name}|Servicio]]\n"
        )

        reply_text = ""
        tool_called: Optional[str] = None
        appointment_created = False
        google_calendar_synced = False
        lower_msg = payload.message.lower()

        # 7. Candidate LLM providers (fast path: priority provider first, no unnecessary slow loops)
        candidates: list[tuple[str, str]] = []
        preferred_provider = (
            payload.llm_provider_override
            or (agent_cfg.llm_provider if agent_cfg else None)
            or "google"
        ).strip().lower()
        preferred_model = (
            payload.llm_model_override
            or (agent_cfg.llm_model if agent_cfg else None)
            or ""
        ).strip()

        if preferred_provider == "google":
            if not preferred_model or "2.0" in preferred_model or "2.5" in preferred_model:
                preferred_model = "gemini-3.8-flash"
            candidates.append(("google", preferred_model))
            if preferred_model != "gemini-flash-latest":
                candidates.append(("google", "gemini-flash-latest"))
            candidates.append(("google", "gemini-pro-latest"))
        elif preferred_provider == "anthropic":
            candidates.append(("anthropic", preferred_model or "claude-3-5-haiku-latest"))
            candidates.append(("google", "gemini-3.8-flash"))
        elif preferred_provider == "openai":
            candidates.append(("openai", preferred_model or "gpt-4o-mini"))
            candidates.append(("google", "gemini-3.8-flash"))
        else:
            candidates.append((preferred_provider, preferred_model))
            candidates.append(("google", "gemini-3.8-flash"))

        llm_errors: list[str] = []
        used_provider: Optional[str] = None

        for pname, pmodel in candidates:
            try:
                provider = LLMProviderFactory().get_provider(pname, pmodel, system_prompt)
                llm_resp = await asyncio.wait_for(
                    provider.generate(
                        system_prompt=system_prompt,
                        messages=history_msgs,
                        temperature=0.3,
                        max_tokens=600,
                    ),
                    timeout=10.0,
                )
                reply_text = (llm_resp.content or "").strip()
                if reply_text:
                    used_provider = f"{pname}/{pmodel or 'default'}"
                    break
                llm_errors.append(f"{pname}/{pmodel}: respuesta vacía")
            except Exception as exc:
                msg = re.sub(r"(sk-|AIza)[A-Za-z0-9_\-]+", "[clave]", str(exc))[:220]
                llm_errors.append(f"{pname}/{pmodel or 'default'}: {type(exc).__name__}: {msg}")
                logger.warning("LLM provider %s/%s unavailable in simulator: %s", pname, pmodel, msg)

        # 8. Run the model's command through the same path as WhatsApp
        if reply_text:
            if any(kw in lower_msg for kw in ("cita", "disponib", "horario", "espacio", "turno", "agendar")):
                tool_called = f"check_availability(date='{tomorrow}')"

            command_type, cmd_args = parse_command(reply_text)
            outcome = await AppointmentService(
                db, org_uuid, GoogleCalendarService(db, org_id_str)
            ).execute_command(
                command_type,
                cmd_args,
                customer_name,
                customer_phone,
                now_co,
                "simulator",
                require_calendar=False,
            )
            google_calendar_synced = outcome.synced
            reply_text = outcome.reply_override or strip_command_tags(reply_text)

            if outcome.changed and command_type == "CANCELAR":
                tool_called = "cancel_appointment()"
            elif outcome.changed:
                appointment_created = True
                action = "reschedule" if command_type == "REAGENDAR" else "book"
                tool_called = (
                    f"{action}_appointment(date='{outcome.booked_start:%Y-%m-%d}', "
                    f"time='{outcome.booked_start:%H:%M}')"
                )

        # 9. Without a model reply nothing can be understood, so nothing is booked
        if not reply_text:
            reply_text = (
                f"¡Hola! Hablas con {agent_name}, asistente virtual de {org_name}. "
                "En este momento no puedo procesar tu mensaje. Por favor intenta de nuevo en unos minutos."
            )

        # 10. Persist assistant reply in PostgreSQL
        try:
            assistant_msg = Message(
                id=uuid.uuid4(),
                conversation_id=active_conv_id,
                organization_id=org_uuid,
                role="assistant",
                content=reply_text,
            )
            db.add(assistant_msg)
            await db.commit()
        except Exception as commit_exc:
            logger.warning("Could not commit assistant reply to DB: %s", commit_exc)
            try:
                await db.rollback()
            except Exception:
                pass

        return SimulateMessageResponse(
            conversation_id=str(active_conv_id),
            reply=reply_text,
            tool_called=tool_called,
            appointment_created=appointment_created,
            google_calendar_synced=google_calendar_synced,
            used_provider=used_provider,
            llm_errors=llm_errors,
        )

    except Exception as top_exc:
        logger.exception("Unexpected error in simulate_conversation_turn: %s", top_exc)
        return SimulateMessageResponse(
            conversation_id=fallback_conv_id,
            reply=(
                f"¡Hola! Soy el asistente virtual de {org_name}. "
                f"¿En qué te puedo colaborar hoy?"
            ),
            tool_called=None,
            appointment_created=False,
            google_calendar_synced=False,
            used_provider=None,
            llm_errors=[f"Endpoint recovery: {type(top_exc).__name__}: {str(top_exc)[:220]}"],
        )
