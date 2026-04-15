# Notes: asyncio vs cats-effect

## The core difference: who yields?

| | asyncio (Python) | cats-effect (Scala) |
|---|---|---|
| Threading model | Single-threaded | Multi-threaded (work-stealing pool) |
| Scheduling | Cooperative | Cooperative + auto-yield |
| Who yields | **You** (`await`) | **Runtime** (every ~512 flatMaps) |
| Forget to yield | Blocks entire event loop | Other fibers still get scheduled |
| Recursive depth | Stack overflows | Safe — trampolined |

---

## asyncio: you are responsible for yielding

The event loop is single-threaded. If you never `await`, nothing else runs.

```python
async def bad():
    while True:        # starves the event loop — nothing else runs
        compute()

async def good():
    while True:
        compute()
        await asyncio.sleep(0)  # yield back to event loop
```

Every `await` is a voluntary yield point. The event loop runs other coroutines
in the gaps.

### Fire-and-forget

```python
asyncio.create_task(write_to_db(evt))  # schedules, does not await
# coroutine runs "later" when event loop gets control
# errors are silently lost unless you attach a callback
```

---

## cats-effect: the runtime yields for you

The `IORuntime` runs on a work-stealing thread pool. It tracks an internal
operation counter per fiber. Every ~512 `flatMap` steps it automatically
inserts a yield (`IO.cede`), regardless of your code.

```scala
// You never write IO.cede — the runtime inserts it automatically
val program = IO.pure(1).flatMap(...).flatMap(...) // runtime yields at ~512
```

### When you'd write IO.cede yourself

Only for opaque CPU loops that the runtime can't see inside:

```scala
IO {
  var i = 0
  while i < 1_000_000 do { work(); i += 1 }  // runtime can't yield here
}

// Fix: bring the loop into IO so the runtime can see it
(0 until 1_000_000).toList.traverse { i =>
  if i % 10_000 == 0 then IO.cede *> IO(work())
  else IO(work())
}
```

### Fire-and-forget

```scala
store.save(evt).start.void   // start = spawn fiber, void = discard Fiber handle
// errors are silently lost — same as asyncio.create_task

// Idiomatic: use Supervisor to capture errors
supervisor.supervise(store.save(evt))
```

---

## Cooperative vs preemptive scheduling

### Cooperative (asyncio, cats-effect within a process)

- Tasks voluntarily yield control
- Safe within a trusted process — no need for locks between yield points
- A single non-yielding task can starve others

### Preemptive (OS process scheduling)

- OS timer interrupts force context switches
- Required for untrusted processes — can't rely on cooperation
- Always need locks because interruption can happen anywhere

### Why cooperative is fine for async I/O

Most async work is **I/O bound** — the task suspends at an `await`/`flatMap`
waiting for network/disk. The yield happens naturally at the I/O boundary.
Preemption is only needed when tasks are CPU-bound and can't be trusted to yield.

### cats-effect and cooperative scheduling

cats-effect uses cooperative scheduling internally but compensates with
auto-yielding — you get the simplicity of cooperative (no arbitrary interruption)
with the fairness of preemptive (no fiber can starve others indefinitely).

---

## The GIL and asyncio

Python 3.13 removed the GIL (optional; 3.13t). This helps **CPU-bound threads**
run truly in parallel. It does **not** change asyncio:

- asyncio is single-threaded by design — the event loop runs on one thread
- Removing the GIL gives that one thread no extra parallelism
- asyncio's advantage is concurrency (many I/O waits interleaved), not parallelism
- For CPU parallelism in Python: `multiprocessing` or `concurrent.futures`

---

## Analogies

| cats-effect | asyncio |
|---|---|
| `IO[A]` | `Coroutine[A]` (unawaited) |
| `unsafeRunSync()` | `asyncio.run()` |
| `flatMap` / `>>=` | `await` |
| `IO.pure(a)` | `async def f(): return a` |
| `parSequence` | `asyncio.gather` |
| `start.void` | `asyncio.create_task` (fire-and-forget) |
| `Fiber` | `asyncio.Task` |
| `IO.cede` | `await asyncio.sleep(0)` |
| `Supervisor` | no direct equivalent |
