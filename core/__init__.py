"""
core package — routing client and shared infrastructure for LLM Cost Autopilot.

Exports:
    send_request  — unified entry point to call any registered model
    ProviderError — custom exception wrapping all provider-level failures
"""

from core.exceptions import ProviderError
from core.router_client import send_request

__all__ = [
    "ProviderError",
    "send_request",
]
