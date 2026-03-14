"""
Row completion — deterministic row-family expansion from EvidencePacks.

Each writer converts an EvidencePack into typed spec rows, applying:
  - Variable family expansion (e.g. AGE → AGE, AGEN, AGEGR, AGEGRN)
  - Source-specific definitions from data_sources/registry.py
  - Provenance linkage from the governing evidence candidate
"""

from .base import RowWriter
from .demographics_writer import DemographicsWriter
from .data_prep_writer import expand_data_prep
from .outcomes_writer import EndpointWriter, CensoringWriter

__all__ = [
    "RowWriter",
    "DemographicsWriter",
    "expand_data_prep",
    "EndpointWriter",
    "CensoringWriter",
]
