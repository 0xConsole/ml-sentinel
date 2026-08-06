"""Base detector protocol."""

from __future__ import annotations

from typing import Protocol

from ..models import Entity, Finding


class Detector(Protocol):
    """Every detector implements this interface."""

    name: str

    def detect(self, client, entity: Entity) -> list[Finding]:
        """Run the detector against ``entity`` and return any findings."""
        ...
