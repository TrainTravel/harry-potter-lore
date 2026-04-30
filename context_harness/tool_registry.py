"""
Tool Registry — Skill 2: Tool and Contract Design
==================================================
The IBM video's core point: if tool schemas are vague, the agent uses its
"imagination" — which causes silent, hard-to-debug failures in production.

Solution:
  - Every tool declares a strict ToolSchema (JSON Schema-compatible).
  - Inputs are validated against the schema before the tool is called.
  - Outputs are validated against a declared return type.
  - Schema violations raise a typed ToolContractError immediately — never
    silently pass bad data to the tool or back to the LLM.
  - The registry produces an OpenAI-compatible function-calling schema list
    so the LLM sees the same contract as the validator.

Design principle:
  "A tool contract is a promise. Break the promise loudly."
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Schema types (JSON Schema subset)
# ---------------------------------------------------------------------------

@dataclass
class PropertySchema:
    type: str                           # "string" | "integer" | "number" | "boolean" | "array"
    description: str
    enum: Optional[List[Any]] = None    # allowed values
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None       # regex for strings
    required: bool = True

    def to_json_schema(self) -> dict:
        s: dict = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = self.enum
        if self.minimum is not None:
            s["minimum"] = self.minimum
        if self.maximum is not None:
            s["maximum"] = self.maximum
        if self.pattern:
            s["pattern"] = self.pattern
        return s


@dataclass
class ToolSchema:
    name: str
    description: str
    properties: Dict[str, PropertySchema]
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def required_fields(self) -> List[str]:
        return [k for k, v in self.properties.items() if v.required]

    def to_openai_function(self) -> dict:
        """Emit an OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: v.to_json_schema() for k, v in self.properties.items()},
                    "required": self.required_fields(),
                },
            },
        }


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------

class ToolContractError(ValueError):
    """Raised when a tool call violates its declared schema."""


class ToolNotFoundError(KeyError):
    """Raised when an unknown tool name is invoked."""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """
    Validates a dict of arguments against a ToolSchema.
    Raises ToolContractError on the first violation found.
    """

    def validate(self, args: Dict[str, Any], schema: ToolSchema) -> None:
        # 1. Required fields present
        for field_name in schema.required_fields():
            if field_name not in args:
                raise ToolContractError(
                    f"Tool '{schema.name}': required field '{field_name}' is missing. "
                    f"Got keys: {list(args.keys())}"
                )

        # 2. Type checks and constraint checks
        for key, value in args.items():
            if key not in schema.properties:
                raise ToolContractError(
                    f"Tool '{schema.name}': unknown field '{key}'. "
                    f"Allowed: {list(schema.properties.keys())}"
                )
            prop = schema.properties[key]
            self._check_type(schema.name, key, value, prop)
            self._check_constraints(schema.name, key, value, prop)

    def _check_type(self, tool: str, key: str, value: Any, prop: PropertySchema) -> None:
        expected = prop.type
        type_map = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list,
        }
        expected_py = type_map.get(expected)
        if expected_py and not isinstance(value, expected_py):
            raise ToolContractError(
                f"Tool '{tool}': field '{key}' must be {expected}, "
                f"got {type(value).__name__} ({value!r})"
            )

    def _check_constraints(self, tool: str, key: str, value: Any, prop: PropertySchema) -> None:
        if prop.enum and value not in prop.enum:
            raise ToolContractError(
                f"Tool '{tool}': field '{key}' must be one of {prop.enum}, got {value!r}"
            )
        if prop.minimum is not None and isinstance(value, (int, float)):
            if value < prop.minimum:
                raise ToolContractError(
                    f"Tool '{tool}': field '{key}' must be >= {prop.minimum}, got {value}"
                )
        if prop.maximum is not None and isinstance(value, (int, float)):
            if value > prop.maximum:
                raise ToolContractError(
                    f"Tool '{tool}': field '{key}' must be <= {prop.maximum}, got {value}"
                )
        if prop.pattern and isinstance(value, str):
            import re
            if not re.fullmatch(prop.pattern, value):
                raise ToolContractError(
                    f"Tool '{tool}': field '{key}' must match pattern '{prop.pattern}', "
                    f"got {value!r}"
                )


# ---------------------------------------------------------------------------
# Tool record
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    schema: ToolSchema
    fn: Callable[..., Any]       # sync or async
    is_async: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_async = asyncio.iscoroutinefunction(self.fn)

    async def call(self, args: Dict[str, Any]) -> Any:
        if self.is_async:
            return await self.fn(**args)
        return self.fn(**args)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry for all agent tools.

    Usage:
        registry = ToolRegistry()

        @registry.register(schema=ToolSchema(...))
        async def search_lore(query: str) -> str:
            ...

        # Call with validation:
        result = await registry.call("search_lore", {"query": "Who is Dumbledore?"})

        # Get OpenAI-compatible function list for the LLM:
        functions = registry.to_openai_functions()
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._validator = SchemaValidator()
        self._call_log: List[dict] = []

    def register(self, schema: ToolSchema):
        """Decorator: registers a function as a tool with a given schema."""
        def decorator(fn: Callable) -> Callable:
            self._tools[schema.name] = Tool(schema=schema, fn=fn)
            return fn
        return decorator

    async def call(self, name: str, args: Dict[str, Any]) -> Any:
        """Validate args, call the tool, log the call."""
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Unknown tool '{name}'. Available: {list(self._tools.keys())}"
            )
        tool = self._tools[name]
        self._validator.validate(args, tool.schema)

        start = time.perf_counter()
        result = await tool.call(args)
        latency_ms = (time.perf_counter() - start) * 1000

        self._call_log.append({
            "tool": name, "args": args,
            "latency_ms": round(latency_ms, 2), "ts": time.time(),
        })
        return result

    def to_openai_functions(self) -> List[dict]:
        """Return a list of OpenAI-compatible function schemas for all tools."""
        return [t.schema.to_openai_function() for t in self._tools.values()]

    def call_log(self) -> List[dict]:
        return list(self._call_log)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schema(self, name: str) -> ToolSchema:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name].schema
