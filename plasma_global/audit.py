"""Stable public interface for simulation and case audits.

Implementation is split by responsibility: result validation, case orchestration,
runtime conservation diagnostics, and immutable report values.
"""

from plasma_global._audit_case import (
    _case_classification as _case_classification,
)
from plasma_global._audit_case import (
    _file_sha256 as _file_sha256,
)
from plasma_global._audit_case import (
    audit_case,
    collect_file_provenance,
)
from plasma_global._audit_result import audit_result
from plasma_global._audit_runtime import runtime_diagnostic_maxima
from plasma_global._audit_types import AuditIssue, AuditReport, CaseAuditReport

__all__ = [
    "AuditIssue",
    "AuditReport",
    "CaseAuditReport",
    "audit_case",
    "audit_result",
    "collect_file_provenance",
    "runtime_diagnostic_maxima",
]
