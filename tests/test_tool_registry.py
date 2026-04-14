"""Tests for context_harness/tool_registry.py — Skill 2"""
import pytest
from context_harness.tool_registry import (
    ToolRegistry, ToolSchema, PropertySchema,
    ToolContractError, ToolNotFoundError, SchemaValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def lore_search_schema() -> ToolSchema:
    return ToolSchema(
        name="search_lore",
        description="Search the HP lore knowledge base",
        properties={
            "query": PropertySchema(type="string", description="Search query", required=True),
            "top_k": PropertySchema(type="integer", description="Number of results",
                                    minimum=1, maximum=20, required=False),
        },
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.register(schema=lore_search_schema())
    async def search_lore(query: str, top_k: int = 5) -> str:
        return f"Results for: {query}"

    return reg


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

def test_validator_passes_valid_args():
    v = SchemaValidator()
    schema = lore_search_schema()
    v.validate({"query": "Who is Dumbledore?"}, schema)   # should not raise


def test_validator_raises_on_missing_required():
    v = SchemaValidator()
    schema = lore_search_schema()
    with pytest.raises(ToolContractError, match="required field 'query'"):
        v.validate({}, schema)


def test_validator_raises_on_wrong_type():
    v = SchemaValidator()
    schema = lore_search_schema()
    with pytest.raises(ToolContractError, match="must be string"):
        v.validate({"query": 42}, schema)


def test_validator_raises_on_unknown_field():
    v = SchemaValidator()
    schema = lore_search_schema()
    with pytest.raises(ToolContractError, match="unknown field"):
        v.validate({"query": "test", "unknown_param": "value"}, schema)


def test_validator_raises_on_minimum_violation():
    v = SchemaValidator()
    schema = lore_search_schema()
    with pytest.raises(ToolContractError, match="must be >= 1"):
        v.validate({"query": "test", "top_k": 0}, schema)


def test_validator_raises_on_maximum_violation():
    v = SchemaValidator()
    schema = lore_search_schema()
    with pytest.raises(ToolContractError, match="must be <= 20"):
        v.validate({"query": "test", "top_k": 100}, schema)


def test_validator_enum_constraint():
    schema = ToolSchema(
        name="set_mode",
        description="Set agent mode",
        properties={
            "mode": PropertySchema(type="string", description="Mode",
                                   enum=["chat", "quiz", "research"], required=True),
        },
    )
    v = SchemaValidator()
    v.validate({"mode": "chat"}, schema)  # valid
    with pytest.raises(ToolContractError, match="must be one of"):
        v.validate({"mode": "invalid"}, schema)


def test_validator_pattern_constraint():
    schema = ToolSchema(
        name="get_char",
        description="Get character by ID",
        properties={
            "char_id": PropertySchema(type="string", description="Character ID",
                                      pattern=r"char-\d+", required=True),
        },
    )
    v = SchemaValidator()
    v.validate({"char_id": "char-42"}, schema)
    with pytest.raises(ToolContractError, match="must match pattern"):
        v.validate({"char_id": "not-valid"}, schema)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

async def test_registry_call_valid(registry):
    result = await registry.call("search_lore", {"query": "Dumbledore"})
    assert "Dumbledore" in result


async def test_registry_call_unknown_tool(registry):
    with pytest.raises(ToolNotFoundError):
        await registry.call("nonexistent_tool", {})


async def test_registry_call_invalid_args_raises_contract_error(registry):
    with pytest.raises(ToolContractError):
        await registry.call("search_lore", {})   # missing required 'query'


async def test_registry_call_log_populated(registry):
    await registry.call("search_lore", {"query": "Voldemort"})
    log = registry.call_log()
    assert len(log) == 1
    assert log[0]["tool"] == "search_lore"
    assert "latency_ms" in log[0]


def test_registry_list_tools(registry):
    assert "search_lore" in registry.list_tools()


def test_registry_to_openai_functions(registry):
    funcs = registry.to_openai_functions()
    assert len(funcs) == 1
    assert funcs[0]["type"] == "function"
    assert funcs[0]["function"]["name"] == "search_lore"
    assert "parameters" in funcs[0]["function"]
    assert "required" in funcs[0]["function"]["parameters"]


def test_registry_get_schema(registry):
    schema = registry.get_schema("search_lore")
    assert schema.name == "search_lore"


def test_schema_required_fields():
    schema = lore_search_schema()
    assert "query" in schema.required_fields()
    assert "top_k" not in schema.required_fields()


def test_property_schema_to_json_schema():
    prop = PropertySchema(type="integer", description="Count", minimum=1, maximum=100)
    js = prop.to_json_schema()
    assert js["type"] == "integer"
    assert js["minimum"] == 1
    assert js["maximum"] == 100
