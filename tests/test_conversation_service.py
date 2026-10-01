import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.conversations.models import Conversation, Message
from app.conversations.services import ConversationService


@pytest.fixture
def mock_whatsapp_client():
    with patch("app.conversations.services.WhatsAppClient") as mock:
        client_instance = mock.return_value
        client_instance.send_text_message = AsyncMock()
        yield client_instance


@pytest.fixture
def mock_llm_factory():
    with patch("app.conversations.services.LLMProviderFactory") as mock:
        factory_instance = mock.return_value
        provider_instance = AsyncMock()

        # Define a mock response object matching the expected structure
        class MockResponse:
            text = "Hello from AI"
            tokens_in = 10
            tokens_out = 5

        provider_instance.generate_response.return_value = MockResponse()
        factory_instance.get_provider.return_value = provider_instance
        yield factory_instance


@pytest.mark.asyncio
async def test_process_incoming_message_creates_conversation(
    db_session: AsyncSession,
    mock_whatsapp_client,
    mock_llm_factory,
):
    """Test the full loop for a new conversation."""
    org_id = uuid.uuid4()

    # Pre-create an AgentConfig to route the message to
    agent_config = AgentConfig(
        organization_id=org_id,
        status="active",
        whatsapp_phone_number_id="phone-123",
        llm_provider="anthropic",
        system_prompt_generated="You are a helpful assistant",
    )
    db_session.add(agent_config)
    await db_session.commit()

    service = ConversationService(db_session)

    # Process message
    await service.process_incoming_message(
        phone_number_id="phone-123",
        customer_phone="5551234567",
        text_content="Hi there",
        customer_name="Alice",
    )

    # 1. Verify DB entities
    stmt = select(Conversation).where(Conversation.customer_phone == "5551234567")
    result = await db_session.execute(stmt)
    conv = result.scalar_one()

    assert conv.organization_id == org_id
    assert conv.customer_name == "Alice"

    stmt = select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)
    result = await db_session.execute(stmt)
    messages = result.scalars().all()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hi there"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hello from AI"

    # 2. Verify WhatsApp Client was called
    mock_whatsapp_client.send_text_message.assert_called_once_with(
        phone_number_id="phone-123",
        to="5551234567",
        text="Hello from AI",
    )

    # 3. Verify LLM Provider was called
    mock_provider = mock_llm_factory.get_provider.return_value
    mock_provider.generate_response.assert_called_once()
