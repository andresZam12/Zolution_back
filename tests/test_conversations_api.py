import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.models import Conversation, Message
from app.conversations.services import ConversationService


@pytest.mark.asyncio
async def test_list_and_get_conversations(db_session: AsyncSession):
    """Test ConversationService list and get methods."""
    org_id = uuid.uuid4()
    service = ConversationService(db_session)

    # 1. Create 2 test conversations
    conv1 = Conversation(
        organization_id=org_id,
        customer_phone="+573100000001",
        customer_name="Paciente Uno",
        status="active",
    )
    conv2 = Conversation(
        organization_id=org_id,
        customer_phone="+573100000002",
        customer_name="Paciente Dos",
        status="active",
    )
    db_session.add_all([conv1, conv2])
    await db_session.flush()

    # 2. Add messages
    msg1 = Message(
        conversation_id=conv1.id,
        organization_id=org_id,
        role="user",
        content="Hola doctor",
    )
    msg2 = Message(
        conversation_id=conv1.id,
        organization_id=org_id,
        role="assistant",
        content="Hola! En que puedo ayudarte?",
    )
    db_session.add_all([msg1, msg2])
    await db_session.commit()

    # 3. List conversations
    items, total = await service.list_conversations(str(org_id), page=1, page_size=10)
    assert total == 2
    assert len(items) == 2

    # 4. Get specific conversation with messages
    fetched = await service.get_conversation(str(org_id), str(conv1.id))
    assert fetched is not None
    assert fetched.customer_name == "Paciente Uno"
    assert len(fetched.messages) == 2
    assert fetched.messages[0].content == "Hola doctor"
    assert fetched.messages[1].content == "Hola! En que puedo ayudarte?"
