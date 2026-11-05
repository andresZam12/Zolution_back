"""
Conversation service layer.
"""

import logging
import time
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.models import AgentConfig
from app.appointments.services import CALENDAR_FAILED_REPLY
from app.conversations.agent import run_agent_turn
from app.conversations.models import Conversation, Message
from app.core.database import bind_tenant, resolve_whatsapp_tenant
from app.integrations.google.calendar import GoogleCalendarService
from app.integrations.whatsapp.client import WhatsAppClient
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)

UNSUPPORTED_MESSAGE_REPLY = (
    "Por ahora solo puedo leer mensajes de texto. ¿Me escribes lo que necesitas, por favor?"
)

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

    async def _active_agent(self, phone_number_id: str) -> tuple[AgentConfig, Organization] | None:
        """Route strictly by the receiving number. Unknown numbers are never assigned to a tenant."""
        organization_id = await resolve_whatsapp_tenant(self.db, phone_number_id)
        if not organization_id:
            logger.warning(f"No AgentConfig linked to WhatsApp phone_number_id: {phone_number_id}")
            return None

        await bind_tenant(self.db, organization_id)
        agent_res = await self.db.execute(
            select(AgentConfig).where(AgentConfig.organization_id == organization_id)
        )
        agent_config = agent_res.scalar_one()
        org_res = await self.db.execute(
            select(Organization).where(Organization.id == organization_id)
        )
        org = org_res.scalar_one_or_none()

        if agent_config.status != "active" or not org or org.status != "active":
            logger.info(f"Ignored WhatsApp message for inactive agent {agent_config.id}")
            return None
        return agent_config, org

    async def _save_incoming(
        self,
        conversation: Conversation,
        text_content: str,
        whatsapp_message_id: str | None,
    ) -> bool:
        """Store the customer's message. Returns False if Meta already delivered it."""
        self.db.add(
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                organization_id=conversation.organization_id,
                whatsapp_message_id=whatsapp_message_id,
                role="user",
                content=text_content,
            )
        )
        try:
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            logger.info(f"Ignored duplicate WhatsApp message: {whatsapp_message_id}")
            return False
        return True

    async def _deliver(self, phone_number_id: str, customer_phone: str, reply: str, conversation_id) -> bool:
        logger.info(f"Sending WhatsApp reply in conversation {conversation_id}")
        try:
            await self.whatsapp_client.send_text_message(
                phone_number_id=phone_number_id,
                to=customer_phone,
                text=reply,
            )
        except Exception as e:
            logger.error(f"Failed to send WhatsApp reply in conversation {conversation_id}: {e}")
            return False
        logger.info(f"WhatsApp reply sent in conversation {conversation_id}")
        return True

    async def process_incoming_message(
        self,
        phone_number_id: str,
        customer_phone: str,
        text_content: str,
        customer_name: str | None,
        whatsapp_message_id: str | None = None,
    ) -> None:
        """Answer a customer's WhatsApp text message."""
        started_at = time.monotonic()
        clean_phone_id = str(phone_number_id).strip()
        logger.info(f"Processing incoming WhatsApp message (phone_id={clean_phone_id})")

        routed = await self._active_agent(clean_phone_id)
        if not routed:
            return
        agent_config, org = routed

        effective_customer_name = (customer_name or "Cliente").strip()
        conversation = await self._get_or_create_conversation(
            agent_config.organization_id, customer_phone, effective_customer_name
        )
        if not await self._save_incoming(conversation, text_content, whatsapp_message_id):
            return

        turn = await run_agent_turn(
            self.db,
            agent_config=agent_config,
            org_name=org.name or "tu negocio",
            conversation_id=conversation.id,
            customer_name=effective_customer_name,
            customer_phone=customer_phone,
            text_content=text_content,
            source="whatsapp",
            calendar=GoogleCalendarService(self.db, str(agent_config.organization_id)),
            llm_factory=self.llm_factory,
            require_calendar=True,
        )

        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            organization_id=agent_config.organization_id,
            role="assistant",
            content=turn.reply,
        )
        self.db.add(assistant_message)
        await self.db.commit()

        delivered = await self._deliver(clean_phone_id, customer_phone, turn.reply, conversation.id)

        # Record the real outcome for the dashboard metrics
        assistant_message.response_time_ms = int((time.monotonic() - started_at) * 1000)
        assistant_message.resolved = (
            turn.model_answered
            and delivered
            and turn.outcome.reply_override != CALENDAR_FAILED_REPLY
        )
        await self.db.commit()

    async def process_unsupported_message(
        self,
        phone_number_id: str,
        customer_phone: str,
        message_type: str,
        customer_name: str | None,
        whatsapp_message_id: str | None = None,
    ) -> None:
        """Tell the customer the agent can't read this kind of message, instead of staying silent."""
        clean_phone_id = str(phone_number_id).strip()
        routed = await self._active_agent(clean_phone_id)
        if not routed:
            return
        agent_config, _org = routed

        conversation = await self._get_or_create_conversation(
            agent_config.organization_id, customer_phone, (customer_name or "Cliente").strip()
        )
        if not await self._save_incoming(conversation, f"[{message_type}]", whatsapp_message_id):
            return

        self.db.add(
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                organization_id=agent_config.organization_id,
                role="assistant",
                content=UNSUPPORTED_MESSAGE_REPLY,
            )
        )
        await self.db.commit()
        await self._deliver(clean_phone_id, customer_phone, UNSUPPORTED_MESSAGE_REPLY, conversation.id)

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
