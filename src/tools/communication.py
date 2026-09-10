"""Phone/messaging tools.

These are simulated — no real telephony/SMS integration — but exercise
the same validate -> execute -> structured result contract as every
other tool, which is the property the agent/safety layer actually cares
about.
"""

from __future__ import annotations

from src.tools.base import ToolResult


def make_phone_call(phone_contact: str) -> ToolResult:
    if not phone_contact:
        return ToolResult.fail("phone_contact must be provided")
    return ToolResult.ok(status="calling", contact=phone_contact)


def send_message(phone_contact: str, message_body: str | None = None) -> ToolResult:
    if not phone_contact:
        return ToolResult.fail("phone_contact must be provided")
    body = message_body or "(no message body provided)"
    return ToolResult.ok(status="sent", contact=phone_contact, message_body=body)
