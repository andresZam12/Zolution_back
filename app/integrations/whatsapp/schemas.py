"""
Pydantic schemas for the WhatsApp Cloud API Webhooks.
"""

from typing import Any

from pydantic import BaseModel, Field


class WhatsAppText(BaseModel):
    body: str


class WhatsAppMessage(BaseModel):
    id: str
    from_: str = Field(alias="from")
    timestamp: str
    type: str
    text: WhatsAppText | None = None


class WhatsAppContactProfile(BaseModel):
    name: str = "Cliente WhatsApp"


class WhatsAppContact(BaseModel):
    profile: WhatsAppContactProfile = Field(default_factory=WhatsAppContactProfile)
    wa_id: str = ""


class WhatsAppValue(BaseModel):
    messaging_product: str = "whatsapp"
    metadata: dict[str, Any] = Field(default_factory=dict)
    contacts: list[WhatsAppContact] | None = None
    messages: list[WhatsAppMessage] | None = None
    statuses: list[dict[str, Any]] | None = None


class WhatsAppChange(BaseModel):
    field: str
    value: WhatsAppValue


class WhatsAppEntry(BaseModel):
    id: str
    changes: list[WhatsAppChange]


class WhatsAppWebhookPayload(BaseModel):
    """
    The root payload sent by Meta to the webhook.
    """

    object: str
    entry: list[WhatsAppEntry]
