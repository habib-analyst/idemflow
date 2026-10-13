"""idemflow — self-diagnosing dead-letter queues.

The dead-letter queue that diagnoses WHY each event failed: retry-signature
cause classification (transient / poison / infra) with auto-remediation.
"""
from .classifier import CLASSES, NaiveBaseline, SignatureClassifier
from .engine import (
    CONNECTION, DOWNSTREAM_5XX, LOGIC_ERROR, MISSING_FIELD, PAYLOAD_TOO_LARGE,
    RATE_LIMIT, SCHEMA_VIOLATION, TIMEOUT, UNKNOWN, Attempt, Engine,
    IdemflowError, RetryHistory, classify_exception,
)
from .remediation import (
    HOLD, QUARANTINE, REQUEUE, PolicyConfig, RemediationDecision,
    RemediationPolicy,
)
from .signatures import FEATURE_NAMES, extract_features, feature_vector
from .simulator import INFRA, POISON, TRANSIENT, generate_dataset, generate_history

__version__ = "0.1.0"
__all__ = [
    "Engine", "RetryHistory", "Attempt", "IdemflowError", "classify_exception",
    "extract_features", "feature_vector", "FEATURE_NAMES",
    "NaiveBaseline", "SignatureClassifier", "CLASSES",
    "RemediationPolicy", "PolicyConfig", "RemediationDecision",
    "REQUEUE", "QUARANTINE", "HOLD",
    "generate_dataset", "generate_history",
    "TRANSIENT", "POISON", "INFRA",
    "TIMEOUT", "RATE_LIMIT", "CONNECTION", "DOWNSTREAM_5XX",
    "SCHEMA_VIOLATION", "MISSING_FIELD", "LOGIC_ERROR",
    "PAYLOAD_TOO_LARGE", "UNKNOWN",
]
