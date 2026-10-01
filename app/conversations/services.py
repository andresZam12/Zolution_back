"""
Conversation service layer.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.models import AgentConfig
from app.conversations.models import Conversation, Message
from app.integrations.whatsapp.client import WhatsAppClient
from app.llm_providers.base import ConversationMessage
from app.llm_providers.factory import LLMProviderFactory

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
                organization_id=organization_id,
                customer_phone=customer_phone,
                customer_name=customer_name,
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
        Main entrypoint for processing an incoming message.
        """
        # 1. Look up AgentConfig by whatsapp_phone_number_id
        # This determines which tenant the message is for.
        stmt = select(AgentConfig).where(AgentConfig.whatsapp_phone_number_id == phone_number_id)
        result = await self.db.execute(stmt)
        agent_config = result.scalar_one_or_none()

        if not agent_config:
            logger.warning(f"No AgentConfig found for phone_number_id: {phone_number_id}")
            return

        if agent_config.status != "active":
            logger.info(f"Agent is not active for organization {agent_config.organization_id}")
            return

        organization_id = str(agent_config.organization_id)

        # 2. Get or create conversation history
        conversation = await self._get_or_create_conversation(
            organization_id, customer_phone, customer_name
        )

        # 3. Append the incoming user message
        user_message = Message(
            conversation_id=conversation.id,
            organization_id=organization_id,
            role="user",
            content=text_content,
        )
        self.db.add(user_message)
        conversation.messages.append(user_message)
        await self.db.flush()

        # 4. Prepare messages for LLM
        # Map our DB messages to the ConversationMessage objects expected by the LLM Provider
        llm_messages = [
            ConversationMessage(role=m.role, content=m.content) for m in conversation.messages
        ]

        # 5. Call the LLM
        try:
            provider = self.llm_factory.get_provider(
                agent_config.llm_provider,
                agent_config.llm_model,
                agent_config.system_prompt_generated or "",
            )
            llm_response = await provider.generate_response(llm_messages)
        except Exception as e:
            logger.error(f"Error calling LLM provider: {e}")
            # Fallback message
            fallback_text = (
                "I'm having a technical issue right now. Please try again in a few minutes."
            )
            llm_response_content = fallback_text
            tokens_in, tokens_out = None, None
        else:
            llm_response_content = llm_response.text
            tokens_in = llm_response.tokens_in
            tokens_out = llm_response.tokens_out

        # 6. Save the assistant response
        assistant_message = Message(
            conversation_id=conversation.id,
            organization_id=organization_id,
            role="assistant",
            content=llm_response_content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        self.db.add(assistant_message)
        await self.db.commit()

        # 7. Send the response back via WhatsApp
        try:
            await self.whatsapp_client.send_text_message(
                phone_number_id=phone_number_id,
                to=customer_phone,
                text=llm_response_content,
            )
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
