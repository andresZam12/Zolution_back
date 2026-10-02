"""
Definitions for LLM Tools (Function Calling).
"""

from typing import Any

# JSON Schema for the tools we provide to the LLM.
# We map these to standard OpenAI-style JSON schema because it is the most
# widely understood format, and the specific providers (Anthropic, Gemini)
# have their own ways of handling them (we'll pass them in appropriately).

AVAILABLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check the business's calendar availability for a given date. Returns a summary of free/busy times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check availability for, in YYYY-MM-DD format (e.g., 2026-10-02).",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a new appointment on the business's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "The full name of the customer booking the appointment.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "The phone number of the customer.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "The start time of the appointment in ISO8601 format (e.g., 2026-10-02T10:00:00Z).",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "The end time of the appointment in ISO8601 format (e.g., 2026-10-02T11:00:00Z).",
                    },
                },
                "required": ["customer_name", "customer_phone", "start_time", "end_time"],
            },
        },
    },
]
