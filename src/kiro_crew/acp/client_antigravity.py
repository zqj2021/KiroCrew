"""Google Antigravity-specific checks for an ``AcpClient`` session.

Accepts both the in-tree ``google-antigravity`` Python SDK ACP adapter
(``sdk-<ver>``) and standalone Google Antigravity ACP 1.x releases (baseline
verified on ``1.4.4``; exact pin enforceable via ``ANTIGRAVITY_STRICT_VERSION_PIN=1``),
while refusing missing or incompatible major versions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kiro_crew.acp.harness.antigravity import (
    ANTIGRAVITY_INSTALL_HINT,
    ANTIGRAVITY_VERIFIED_VERSION,
    is_accepted_antigravity_version,
)
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.agent_sdk.backends import ACP_BACKEND_ANTIGRAVITY

if TYPE_CHECKING:
    from kiro_crew.acp.client import AcpClient


def _refuse_unverified_antigravity_version(self: AcpClient) -> None:
    """Refuse an incompatible or unverified Antigravity ACP runtime."""
    if self.backend == ACP_BACKEND_ANTIGRAVITY:
        version = self.agent_version
        if is_accepted_antigravity_version(version):
            return
        what = (
            f"Google Antigravity ACP {version} is not a supported or verified version "
            f"(baseline {ANTIGRAVITY_VERIFIED_VERSION}), so this chat was refused."
            if version
            else "Google Antigravity ACP did not report its version, so Kiro Crew could "
            f"not confirm compatibility ({ANTIGRAVITY_VERIFIED_VERSION}) "
            "and refused this chat."
        )
        raise AcpToolGateUnroutable(
            f"{what} {ANTIGRAVITY_INSTALL_HINT} Then start a new chat.",
            transient=False,
            user_worded=True,
        )
