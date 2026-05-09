# Agent-Driven Development Playbook

Based on Gabor Mayer's 21-agent framework (article + github.com/gabormayer/pm).
Adapted for the Harry Potter Lore Agent project.

> **Caveat on agent names:** Mayer explicitly names System Analyst, Spaghetti Agent,
> UX Flow Architect, Designer, CTO, Test Architect, and Product Council. The rest of
> the roster below is reconstructed from the four-cluster structure and standard
> software team roles — treat unattributed names as approximations until you verify
> against his Maven course or GitHub.

---

## The Four Clusters

```
┌─────────────────────┐  ┌─────────────────────┐
│   STRATEGY          │  │   DESIGN / UX        │
│   CTO               │  │   UX Flow Architect  │
│   Product Council   │  │   Designer           │
│   System Analyst    │  │   Content Strategist │
│   Market Analyst    │  │   Accessibility Lead │
└─────────────────────┘  └─────────────────────┘
┌─────────────────────┐  ┌─────────────────────┐
│   ENGINEERING       │  │   QUALITY            │
│   Architect         │  │   Test Architect     │
│   Backend Engineer  │  │   Spaghetti Agent    │
│   Frontend Engineer │  │   Security Reviewer  │
│   Data Engineer     │  │   Performance Eng.   │
│   DevOps Engineer   │  │   Docs Reviewer      │
└─────────────────────┘  └─────────────────────┘
```

---

## The Four Gotchas

These are Mayer's explicit warnings — the failure modes that kill agent-driven projects.

**1. Agents lie confidently.**
They'll say "done" when the task is half-finished, "tested" when no test ran,
"no breaking changes" when there are. Always verify outputs against the actual
artefact (file diff, test result, running server), not the agent's summary.

**2. Context window rot.**
A long agent chain degrades. By agent 6–8, early decisions are forgotten or
contradicted. Mitigation: summarise each cluster's output into a short brief
before handing to the next cluster. Don't pass the full conversation.

**3. The Spaghetti Agent must run last (or separately).**
If you ask a creative/implementation agent to also find its own edge cases,
it will defend its own work. The Spaghetti Agent's value is adversarial —
it must be seeded with the output, not the thought process.

**4. Product Council is not a rubber stamp.**
If you ask the Product Council to review something you've already built,
confirmation bias kicks in. Run it *before* implementation to surface
contradictions in requirements, not after to feel validated.

---

## Agent Prompt Templates

These are direct from Mayer's article, lightly adapted for this project.

### System Analyst
```
You are a senior system analyst. Your job is to translate business goals
into unambiguous technical requirements.

Input: [feature description or user story]

Produce:
- Acceptance criteria (numbered, testable, no ambiguity)
- Data contracts (inputs, outputs, types)
- Edge cases that must be handled
- Explicit non-goals (what this feature does NOT do)

Do not suggest implementation. Do not write code.
```

### CTO / Technical Decision
```
You are a CTO reviewing a proposed technical approach for a production system.

Context: [the proposed solution]
Constraints: [existing architecture, team, timeline]

Evaluate:
- Does this introduce unacceptable coupling or tech debt?
- What breaks in 6 months if this ships?
- What's the simplest approach that meets the requirements?
- What would you do differently and why?

Be direct. Disagreement is more useful than agreement.
```

### Spaghetti Agent
```
You are a hostile QA engineer. Your job is to break things.

Given: [feature description + implementation summary]

Find:
- Inputs that crash or corrupt state
- Race conditions or async hazards
- Assumptions that will be wrong in production
- The scenario the developer definitely didn't test

For each finding: state the input, the expected failure mode,
and the severity (data loss / incorrect output / degraded UX).
Do not suggest fixes. Just find the holes.
```

### Test Architect
```
You are a test architect. Your job is to define the testing strategy,
not write the tests.

Given: [feature or module description]

Produce:
- Test pyramid allocation (unit / integration / E2E counts)
- What the metric for each layer is (what does "passing" mean?)
- Which cases must be deterministic vs. which use real LLM calls
- What DummyLM answer templates are needed
- The SLO threshold if this is an LLM-graded output

One sentence per decision. No test code.
```

### Product Council
```
You are a panel of three reviewers: a skeptical user, a product manager,
and an engineer who has to maintain this forever.

Requirement under review: [feature spec]

Each reviewer responds in character:
- User: "What I actually need is..." / "This confuses me because..."
- PM: "The success metric is unclear on..." / "This conflicts with..."
- Engineer: "In 6 months this breaks when..." / "The hidden assumption is..."

Surface contradictions between the three. Do not resolve them — that's the
author's job.
```

### UX Flow Architect
```
You are a UX flow architect. You think in states and transitions,
not screens.

Given: [user goal and current interaction model]

Map:
- The happy path (numbered steps, one action per step)
- Every branch point where the user could go wrong
- Every state that requires a loading/error/empty treatment
- The one place where users will drop off and why

No wireframes. No copy. States and transitions only.
```

---

## Applying This to the HP Lore Agent

### When to use which cluster

| Scenario | Start with | Then |
|---|---|---|
| New DSPy mode | System Analyst → Test Architect | Engineering → Spaghetti Agent |
| API endpoint change | System Analyst → Product Council | Engineering → Test Architect |
| New corpus / RAG change | System Analyst → Data Engineer | Test Architect → Spaghetti Agent |
| UI feature (Lovable) | UX Flow Architect → Designer | Engineering → Spaghetti Agent |
| Architecture decision | CTO → System Analyst | Product Council |
| Performance/reliability | Spaghetti Agent first | Engineering → Test Architect |

---

### Workflow: Adding a New DSPy Mode

This is the most common task in this project. The checklist from CLAUDE.md
already captures the engineering steps — this wraps it in the agent framework.

**Step 1 — System Analyst**
Prompt with: the mode name, what it does, the user need it serves.
Output: acceptance criteria, input/output field contracts, explicit non-goals.

**Step 2 — Test Architect**
Prompt with: System Analyst output.
Output: how many TRAINSET examples, what the metric checks, DummyLM template
fields, the SLO threshold, what a compile smoke test must assert.

**Step 3 — Product Council** *(run before writing any code)*
Prompt with: System Analyst output.
Surface: does this conflict with existing modes? Is the intent router going to
misclassify it? Does the name (`guided_learning`, `open_analysis`) mean what
users expect?

**Step 4 — Engineering**
Now implement. Follow CLAUDE.md checklist:
- Signature → Module → Trainset → Metric → slo_check wiring → compile smoke test

**Step 5 — Spaghetti Agent**
Prompt with: the Signature, the metric, and 2 example EVALSET questions.
It will find: input strings that confuse the intent router, edge cases the
metric doesn't catch, multi-turn degradation the Signature doesn't handle.

**Step 6 — Verify against artefacts**
Don't trust agent summaries. Run:
```bash
python -m pytest tests/test_compile_smoke.py -k <mode>
python -m evals.slo_check --mode <mode> --verbose
```

---

### Workflow: Knowledge Graph RAG (Next Feature)

The planned `LoreGraph` feature (NetworkX graph over character_lore.jsonl,
graph traversal alongside vector retrieval) is a good candidate to run through
the full pipeline before implementing.

**System Analyst prompt seed:**
```
Feature: Knowledge Graph RAG for the HP Lore Agent.

Goal: answer multi-hop relationship queries (e.g. "who owned the Elder Wand
and in what order?") that require traversing connected entities, not just
vector-similar passages.

Current state: deep_research uses MultiToolDeepResearchModule (vector_search,
character_search, topic_search via ToolRegistry). No graph layer exists.

Existing corpus: 32 HP lore docs in ChromaDB + character_lore collection.
character_lore.jsonl has character-level chunks with metadata.

Constraints: must not break existing deep_research EVALSET (SLO 80%).
Must not require a separate service. Must work with the existing ToolRegistry
pattern (call_sync, ToolSchema).
```

Run System Analyst → CTO → Test Architect before writing a line of code.

---

## Prompt Chaining Rules (Mayer's)

1. **Each agent gets a clean context.** Copy-paste the relevant output from the
   prior agent into a new conversation. Don't continue the same thread.

2. **Compress between clusters.** Before moving from Strategy → Engineering,
   write a 5-bullet brief of what Strategy decided. This is what you hand to
   Engineering, not the full Strategy transcript.

3. **Run Spaghetti Agent in a fresh session** with no prior context from
   the implementation. Seed it with the spec and a brief description of what
   was built. Adversarial agents are less adversarial when they've watched
   you build the thing.

4. **Product Council runs on specs, not code.** If you show it working code
   it will find reasons why the code is fine. Show it requirements.

---

## Project-Specific Agent Prompts

### For DSPy Signature review
```
You are a DSPy expert reviewing a new Signature for correctness.

Signature: [paste Signature class]

Check:
- Are InputField and OutputField descriptions precise enough for a LLM to
  follow without ambiguity?
- Will ChainOfThought's reasoning field conflict with any OutputField name?
- Is `context` always an InputField (never an OutputField)?
- Will BootstrapFewShot's teacher LLM be able to generate demos that pass
  the metric, or is the metric too strict for the teacher to satisfy?
- Does the Signature have a chat_history field if this is a multi-turn mode?
  Is the multi-turn behaviour fully specified in the field description?
```

### For SLO / EVALSET review
```
You are a test architect reviewing a held-out evaluation set.

EVALSET: [paste EVALSET examples]
Metric: [paste metric function]
SLO: [threshold]%

Check:
- Is the EVALSET disjoint from TRAINSET on (primary_input, output) pairs?
- Does it cover the full difficulty range the mode will see in production?
- Is the metric's pass condition achievable by a model that genuinely
  understands the question, without being gameable by a model that doesn't?
- Is the SLO threshold set at "good enough to ship" or "too easy to be useful"?
- Are there multi-hop or multi-turn examples if the mode supports them?
```

### For ToolRegistry contract review
```
You are reviewing a ToolSchema for correctness and safety.

ToolSchema: [paste schema]

Check:
- Are minimum/maximum bounds tight enough to prevent misuse?
- Does the description tell the LLM exactly when to use this tool vs the others?
- Will SchemaValidator catch the most likely LLM output errors?
- Is there a fallback if this tool fails mid-chain?
- Does the tool name conflict with any existing registered tool?
```

---

## Maintenance Notes

- Update the **Spaghetti Agent prompt** whenever a new mode is added — it
  should know all valid modes so it can probe misrouting.
- Run the **Product Council prompt** before any change to the intent router,
  since the router's classification boundary affects all modes.
- The **Test Architect prompt** should be re-run whenever the SLO thresholds
  are changed — the threshold is a product decision, not just a number.
