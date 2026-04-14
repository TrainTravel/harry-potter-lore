"""
Security Layer — Skill 5: Security and Safety
=============================================
The IBM video's point: agents are attack surfaces. Prompt injection, data
exfiltration, and policy violations are real production risks.

This module implements:
  - PromptGuard    — detects prompt injection patterns in user input
  - InputValidator — enforces length limits, character allowlists, content rules
  - OutputFilter   — scans LLM output for policy violations before returning
  - PermissionBoundary — declares which tools a given capability may call

Design principle:
  "Validate at the boundary. Trust nothing that arrives from outside."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Threat levels
# ---------------------------------------------------------------------------

class ThreatLevel(str, Enum):
    SAFE    = "safe"
    WARN    = "warn"     # suspicious but not certain
    BLOCK   = "block"    # definite violation — reject immediately


@dataclass
class SecurityResult:
    level: ThreatLevel
    reason: str = ""
    matched_pattern: str = ""

    @property
    def is_safe(self) -> bool:
        return self.level == ThreatLevel.SAFE

    @property
    def should_block(self) -> bool:
        return self.level == ThreatLevel.BLOCK


# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

# Canonical prompt injection signatures — ordered by confidence.
# Each tuple: (regex_pattern, threat_level, human_readable_reason)
_INJECTION_PATTERNS: List[Tuple[str, ThreatLevel, str]] = [
    # Classic role override
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", ThreatLevel.BLOCK,
     "Role override: 'ignore previous instructions'"),
    (r"disregard\s+(all\s+)?(previous|prior)\s+instructions?", ThreatLevel.BLOCK,
     "Role override: 'disregard instructions'"),
    (r"you\s+are\s+now\s+(a|an|the)\s+\w+", ThreatLevel.WARN,
     "Persona override: 'you are now ...'"),

    # System prompt extraction attempts
    (r"(repeat|print|output|reveal|show|tell me)\s+(your|the)\s+(system\s+)?prompt", ThreatLevel.BLOCK,
     "Prompt extraction attempt"),
    (r"what\s+(are|were)\s+your\s+(original\s+)?instructions", ThreatLevel.BLOCK,
     "Instruction extraction attempt"),

    # Jailbreak keywords
    (r"\b(DAN|STAN|jailbreak|developer\s+mode|god\s+mode)\b", ThreatLevel.BLOCK,
     "Known jailbreak keyword"),

    # Context escape sequences
    (r"```\s*(system|assistant|human|user)\s*\n", ThreatLevel.WARN,
     "Context escape via code block"),
    (r"<\s*/?(system|assistant|human|user)\s*>", ThreatLevel.WARN,
     "Context escape via XML-like tag"),

    # Indirect injection (content that tells the model to do something)
    (r"if\s+you\s+(are|were)\s+asked.{0,60}(do not|don't|never)", ThreatLevel.WARN,
     "Conditional instruction injection"),
]

_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), level, reason)
    for p, level, reason in _INJECTION_PATTERNS
]


class PromptGuard:
    """
    Scans user input for prompt injection signatures.

    Usage:
        guard = PromptGuard()
        result = guard.scan("Ignore all previous instructions and ...")
        if result.should_block:
            raise SecurityError(result.reason)
    """

    def scan(self, text: str) -> SecurityResult:
        """Return the highest-threat match found, or SAFE."""
        highest = SecurityResult(ThreatLevel.SAFE)

        for pattern, level, reason in _COMPILED_PATTERNS:
            match = pattern.search(text)
            if match:
                result = SecurityResult(
                    level=level,
                    reason=reason,
                    matched_pattern=match.group(0),
                )
                # BLOCK beats WARN; return immediately on BLOCK
                if level == ThreatLevel.BLOCK:
                    return result
                if level == ThreatLevel.WARN:
                    highest = result

        return highest


# ---------------------------------------------------------------------------
# Input Validator
# ---------------------------------------------------------------------------

@dataclass
class InputValidator:
    """
    Enforces structural constraints on user input before it enters the pipeline.

    These checks catch simple misuse without needing an LLM call.
    """

    max_length: int = 4096
    min_length: int = 1
    # If set, only these characters are allowed (regex character class body)
    allowed_chars_pattern: Optional[str] = None
    # Block inputs that match any of these patterns (e.g. SQL injection)
    blocked_patterns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._blocked = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]

    def validate(self, text: str) -> SecurityResult:
        if len(text) < self.min_length:
            return SecurityResult(ThreatLevel.BLOCK, f"Input too short (min {self.min_length})")

        if len(text) > self.max_length:
            return SecurityResult(ThreatLevel.BLOCK, f"Input too long (max {self.max_length} chars)")

        if self.allowed_chars_pattern:
            if not re.fullmatch(f"[{self.allowed_chars_pattern}]+", text, re.DOTALL):
                return SecurityResult(ThreatLevel.BLOCK, "Input contains disallowed characters")

        for pat in self._blocked:
            m = pat.search(text)
            if m:
                return SecurityResult(ThreatLevel.BLOCK,
                                      f"Blocked pattern matched: {m.group(0)!r}")

        return SecurityResult(ThreatLevel.SAFE)


# ---------------------------------------------------------------------------
# Output Filter
# ---------------------------------------------------------------------------

# Patterns that should never appear in agent output
_OUTPUT_VIOLATIONS: List[Tuple[str, str]] = [
    # PII leakage
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern detected in output"),
    (r"\b(?:\d[ -]?){13,16}\b", "Credit card number pattern detected"),
    # Credential leakage
    (r"(api[_-]?key|secret|password|token)\s*[:=]\s*\S+", "Credential pattern in output"),
    # System prompt leakage
    (r"(my system prompt|my instructions are|i was told to)", "Potential system prompt leak"),
    # Harmful content signals (extend as needed)
    (r"\b(make\s+a\s+bomb|synthesize\s+\w+\s+drug)\b", "Harmful content policy violation"),
]

_COMPILED_OUTPUT = [
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in _OUTPUT_VIOLATIONS
]


@dataclass
class OutputFilter:
    """
    Scans LLM output before it reaches the user.

    If a violation is found the caller should replace the output with
    a safe fallback message rather than returning the raw LLM text.
    """

    extra_patterns: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._extra = [
            (re.compile(p, re.IGNORECASE), r) for p, r in self.extra_patterns
        ]

    def scan(self, text: str) -> SecurityResult:
        for pattern, reason in _COMPILED_OUTPUT + self._extra:
            if pattern.search(text):
                return SecurityResult(ThreatLevel.BLOCK, reason)
        return SecurityResult(ThreatLevel.SAFE)

    def safe_fallback(self) -> str:
        return "I'm unable to provide that response due to a content policy restriction."


# ---------------------------------------------------------------------------
# Permission Boundary
# ---------------------------------------------------------------------------

@dataclass
class PermissionBoundary:
    """
    Declares which tools each capability is allowed to call.

    Prevents a compromised or misbehaving capability from calling tools
    outside its declared scope (e.g. a quiz capability shouldn't be able
    to call a file-write tool).

    Usage:
        boundary = PermissionBoundary(rules={
            "chat":     {"search_lore", "get_character"},
            "quiz":     {"search_lore"},
            "admin":    {"search_lore", "ingest_document", "delete_document"},
        })
        boundary.check("chat", "delete_document")  # raises PermissionDeniedError
    """

    rules: Dict[str, Set[str]] = field(default_factory=dict)
    default_allow: bool = False   # if True, unknown capabilities can call anything

    def check(self, capability: str, tool_name: str) -> None:
        allowed = self.rules.get(capability)
        if allowed is None:
            if not self.default_allow:
                raise PermissionDeniedError(
                    f"Capability '{capability}' has no tool permissions defined."
                )
            return
        if tool_name not in allowed:
            raise PermissionDeniedError(
                f"Capability '{capability}' is not permitted to call tool '{tool_name}'. "
                f"Allowed: {sorted(allowed)}"
            )

    def allowed_tools(self, capability: str) -> Set[str]:
        return set(self.rules.get(capability, set()))


class PermissionDeniedError(PermissionError):
    """Raised when a capability attempts to call an unpermitted tool."""


# ---------------------------------------------------------------------------
# Security pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

class SecurityPipeline:
    """
    Runs all security checks in order for one request/response cycle.

    Usage:
        pipeline = SecurityPipeline()
        pipeline.check_input(user_message)     # raises SecurityError on block
        pipeline.check_output(llm_response)    # raises SecurityError on block
    """

    def __init__(
        self,
        guard: Optional[PromptGuard] = None,
        validator: Optional[InputValidator] = None,
        output_filter: Optional[OutputFilter] = None,
    ) -> None:
        self._guard = guard or PromptGuard()
        self._validator = validator or InputValidator()
        self._output_filter = output_filter or OutputFilter()

    def check_input(self, text: str) -> None:
        result = self._validator.validate(text)
        if result.should_block:
            raise SecurityError(f"Input rejected: {result.reason}")

        result = self._guard.scan(text)
        if result.should_block:
            raise SecurityError(f"Injection detected: {result.reason}")

    def check_output(self, text: str) -> str:
        result = self._output_filter.scan(text)
        if result.should_block:
            return self._output_filter.safe_fallback()
        return text


class SecurityError(ValueError):
    """Raised when input or output fails a security check."""
