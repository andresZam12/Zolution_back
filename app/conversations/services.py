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
        from app.integrations.google.calendar import GoogleCalendarService
        from app.llm_providers.tools import AVAILABLE_TOOLS

        try:
            provider = self.llm_factory.get_provider(
                agent_config.llm_provider,
                agent_config.llm_model,
                agent_config.system_prompt_generated or "",
            )
            llm_response = await provider.generate(
                system_prompt=agent_config.system_prompt_generated or "",
                messages=llm_messages,
                tools=AVAILABLE_TOOLS,
            )

            # If the LLM requested a tool call, execute it
            if getattr(llm_response, "tool_calls", None):
                # We only process the first tool call in this MVP for simplicity
                tool_call = llm_response.tool_calls[0]
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"].get("arguments", "{}")
                import json

                logger.info(f"LLM requested tool call: {function_name} with args {arguments}")

                try:
                    args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args_dict = {}

                calendar_service = GoogleCalendarService(self.db, organization_id)
                tool_result = "Tool execution failed."

                try:
                    if function_name == "check_availability":
                        date_str = args_dict.get("date")
                        if date_str:
                            tool_result = await calendar_service.get_availability(date_str)
                    elif function_name == "book_appointment":
                        tool_result = await calendar_service.create_appointment(
                            customer_name=args_dict.get("customer_name", ""),
                            customer_phone=args_dict.get("customer_phone", ""),
                            start_time=args_dict.get("start_time", ""),
                            end_time=args_dict.get("end_time", ""),
                        )
                except Exception as e:
                    logger.error(f"Error executing tool {function_name}: {e}")
                    tool_result = f"Error: {e}"

                # Append assistant's tool call request and the tool's result to the history
                llm_messages.append(
                    ConversationMessage(role="assistant", content="", tool_calls=[tool_call])
                )
                llm_messages.append(
                    ConversationMessage(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                        name=function_name,
                    )
                )

                # Call LLM again to get the final text response
                llm_response = await provider.generate(
                    system_prompt=agent_config.system_prompt_generated or "",
                    messages=llm_messages,
                )

        except Exception as e:
            logger.exception(f"Error calling LLM provider: {e}")
            # Fallback message
            fallback_text = (
                "I'm having a technical issue right now. Please try again in a few minutes."
            )
            llm_response_content = fallback_text
            tokens_in, tokens_out = None, None
        else:
            llm_response_content = llm_response.content
            tokens_in = llm_response.input_tokens
            tokens_out = llm_response.output_tokens

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
