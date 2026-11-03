"""
Pydantic schemas for the WhatsApp Cloud API Webhooks.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WhatsAppBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class WhatsAppText(WhatsAppBaseModel):
    body: str


class WhatsAppMessage(WhatsAppBaseModel):
    id: str
    from_: str = Field(alias="from")
    timestamp: str
    type: str
    text: WhatsAppText | None = None


class WhatsAppContactProfile(WhatsAppBaseModel):
    name: str = "Cliente WhatsApp"


class WhatsAppContact(WhatsAppBaseModel):
    profile: WhatsAppContactProfile = Field(default_factory=WhatsAppContactProfile)
    wa_id: str = ""


class WhatsAppValue(WhatsAppBaseModel):
    messaging_product: str = "whatsapp"
    metadata: dict[str, Any] = Field(default_factory=dict)
    contacts: list[WhatsAppContact] | None = None
    messages: list[WhatsAppMessage] | None = None
    statuses: list[dict[str, Any]] | None = None


class WhatsAppChange(WhatsAppBaseModel):
    field: str
    value: WhatsAppValue


class WhatsAppEntry(WhatsAppBaseModel):
    id: str
    changes: list[WhatsAppChange]


class WhatsAppWebhookPayload(WhatsAppBaseModel):
    """
    The root payload sent by Meta to the webhook.
    """

    object: str
    entry: list[WhatsAppEntry]
