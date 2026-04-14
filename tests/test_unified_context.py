"""Tests for context_harness/unified_context.py — Skills 1 and 7"""
import asyncio
import pytest
from context_harness.unified_context import (
    UnifiedContext, ContextBuilder, EscalationPolicy,
    SessionStore, TurnRuntimeManager, TurnStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return SessionStore(db_path=str(tmp_path / "sessions.db"))


@pytest.fixture
def policy():
    return EscalationPolicy(escalate_on_empty_retrieval=True, max_consecutive_failures=3)


# ---------------------------------------------------------------------------
# UnifiedContext
# ---------------------------------------------------------------------------

def test_unified_context_to_messages_order():
    ctx = UnifiedContext(
        session_id="s1", turn_id="t1", user_message="Who is Dumbledore?",
        summary_context="Earlier we discussed Hogwarts.",
        retrieval_context="Dumbledore is headmaster.",
        memory_context="User prefers concise answers.",
    )
    msgs = ctx.to_messages()
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"     # summary first
    assert roles[-1] == "user"      # user message last
    assert "user" in roles
    assert "system" in roles


def test_unified_context_empty_slots_omitted():
    ctx = UnifiedContext(session_id="s1", turn_id="t1", user_message="Hello")
    msgs = ctx.to_messages()
    # Only user message when all slots are empty
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_unified_context_token_estimate_positive():
    ctx = UnifiedContext(
        session_id="s1", turn_id="t1",
        user_message="What are the Deathly Hallows?",
        retrieval_context="The Elder Wand the Resurrection Stone and the Cloak.",
    )
    assert ctx.token_estimate() > 0


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

async def test_session_created_on_first_access(store):
    await store.get_or_create_session("sess-1")
    await store.get_or_create_session("sess-1")   # idempotent
    # Should not raise


async def test_save_and_retrieve_messages(store):
    await store.get_or_create_session("sess-2")
    await store.save_message("sess-2", "user", "Hello")
    await store.save_message("sess-2", "assistant", "Hi there!")
    msgs = await store.recent_messages("sess-2")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


async def test_recent_messages_limit(store):
    await store.get_or_create_session("sess-3")
    for i in range(10):
        await store.save_message("sess-3", "user", f"Message {i}")
    msgs = await store.recent_messages("sess-3", limit=3)
    assert len(msgs) == 3


async def test_turn_open_and_close(store):
    await store.get_or_create_session("sess-4")
    turn_id = await store.open_turn("sess-4", "chat")
    assert turn_id
    await store.close_turn(turn_id, TurnStatus.COMPLETED)   # should not raise


# ---------------------------------------------------------------------------
# EscalationPolicy
# ---------------------------------------------------------------------------

def make_ctx(user_message: str, retrieval: str = "") -> UnifiedContext:
    return UnifiedContext(
        session_id="s", turn_id="t",
        user_message=user_message,
        retrieval_context=retrieval,
    )


def test_escalate_on_trigger_phrase(policy):
    ctx = make_ctx("I want to speak to a human please")
    assert policy.should_escalate(ctx) is True


def test_escalate_on_empty_retrieval_for_question(policy):
    ctx = make_ctx("Who is Voldemort?", retrieval="")
    assert policy.should_escalate(ctx) is True


def test_no_escalate_when_retrieval_present(policy):
    ctx = make_ctx("Who is Voldemort?", retrieval="Voldemort is the Dark Lord.")
    assert policy.should_escalate(ctx) is False


def test_no_escalate_normal_message(policy):
    ctx = make_ctx("Thanks for the answer!", retrieval="")
    assert policy.should_escalate(ctx) is False


def test_escalate_on_error_type(policy):
    ctx = make_ctx("Hello", retrieval="some context")
    err = PermissionError("not allowed")
    assert policy.should_escalate(ctx, error=err) is True


def test_escalation_message_is_string(policy):
    ctx = make_ctx("tell me something")
    msg = policy.escalation_message(ctx)
    assert isinstance(msg, str)
    assert len(msg) > 0


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

async def test_context_builder_assembles_history(store, tmp_path):
    builder = ContextBuilder(store, history_limit=5)
    await store.get_or_create_session("sess-5")
    await store.save_message("sess-5", "user", "Previous question")
    ctx = await builder.build("sess-5", "turn-1", "New question", capability="chat")
    assert ctx.session_id == "sess-5"
    assert ctx.user_message == "New question"
    assert len(ctx.history_context) >= 1


async def test_context_builder_no_rag_returns_empty_retrieval(store):
    builder = ContextBuilder(store, rag_pipeline=None, assembler=None)
    await store.get_or_create_session("sess-6")
    ctx = await builder.build("sess-6", "t1", "Who is Harry?")
    assert ctx.retrieval_context == ""
