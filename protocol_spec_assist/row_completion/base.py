"""
Base class for row writers.

A RowWriter converts an EvidencePack into a list of VariableRow instances
(or ImportantDate / TimePeriod for Data Prep). It applies:
  1. Deterministic row-family expansion (e.g. AGE → AGE, AGEN)
  2. Source-specific definitions from data_sources/registry.py
  3. Provenance linkage from the governing evidence candidate
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from ..schemas.evidence import EvidencePack
from ..spec_output.spec_schema import VariableRow


class RowWriter(ABC):
    """Abstract base for all row writers."""

    # Subclasses set these
    concept: str = ""            # concept key in packs dict
    target_field: str = ""       # attribute name on ProgramSpec (e.g. "demographics")

    @abstractmethod
    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        """Given an evidence pack, return all rows for the spec tab.

        Args:
            pack: EvidencePack from the concept finder.
            data_source: Data source key for source-specific definitions.

        Returns:
            List of VariableRow instances ready for the spec tab.
        """
        ...

    def _get_governing_candidate(self, pack: EvidencePack):
        """Get the governing candidate: reviewer-selected or top-ranked."""
        if pack.selected_candidate is not None:
            return pack.selected_candidate
        if pack.candidates:
            return pack.candidates[0]
        return None

    def _get_all_candidates(self, pack: EvidencePack):
        """Get all relevant candidates: reviewer-selected or all."""
        if pack.selected_candidates is not None:
            return pack.selected_candidates
        return pack.candidates
