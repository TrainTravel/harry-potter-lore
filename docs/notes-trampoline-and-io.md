# Notes: Trampolining and the IO Monad

## The stack overflow problem

Direct recursion allocates a JVM stack frame for every call. The JVM stack
is limited (~10k frames by default). Deep recursion causes `StackOverflowError`.

```scala
def sum(n: Long): Long =
  if n <= 0 then 0
  else n + sum(n - 1)   // NOT tail position — frame kept alive for `+ n`
                         // stack grows with every call
```

The frame can't be freed because it still has work to do (`+ n`) after the
recursive call returns.

---

## Tail recursion: the call IS the last thing

If the recursive call is in tail position — nothing happens after it — the JVM
(with `@tailrec`) can reuse the same frame:

```scala
@tailrec
def sum(n: Long, acc: Long = 0): Long =
  if n <= 0 then acc
  else sum(n - 1, acc + n)   // tail position — frame can be reused
```

The accumulator carries the pending work instead of the stack.

---

## The trampoline pattern

When you can't express recursion as tail-recursive directly, a trampoline
moves the pending work off the stack and onto the heap.

### The data type

```scala
enum Trampoline[A]:
  case Done(value: A)
  case More(next: () => Trampoline[A])  // "bounce me again"
```

### The recursive function returns data, not a result

```scala
def sum(n: Long, acc: Long = 0): Trampoline[Long] =
  if n <= 0 then Done(acc)
  else More(() => sum(n - 1, acc + n))  // lambda — sum(n-1) not called yet
```

### The run() loop is the only recursive thing — and it IS tail-recursive

```scala
@tailrec
def run[A](t: Trampoline[A]): A =
  t match
    case Done(a)    => a
    case More(next) => run(next())   // tail position — always
```

`run` never accumulates frames because it has nothing to do after the recursive
call. All pending work is encoded in the next `More` thunk on the heap.

---

## Why it doesn't stackoverflow: the key insight

```
Direct recursion:   pending work lives on the STACK (frame kept alive)
Trampolining:       pending work lives on the HEAP (lambda stored in More node)
```

The stack depth of `run` is always 1. The chain of `More` nodes can be
arbitrarily long — it's just heap-allocated data.

---

## IO is a trampolined monad

`IO`'s internal representation is essentially a sealed ADT — a large nested
data structure built on the heap:

```scala
sealed trait IO[A]
case class Pure[A](value: A)                       extends IO[A]
case class Delay[A](thunk: () => A)                extends IO[A]
case class FlatMap[A, B](io: IO[A], f: A => IO[B]) extends IO[B]
```

When you write:

```scala
def sum(n: Long): IO[Long] =
  if n <= 0 then IO.pure(0)
  else IO.pure(n).flatMap(x => sum(n - 1).map(_ + x))
```

`sum(1_000_000)` builds this tree on the heap:

```
FlatMap(
  Pure(1_000_000),
  x => FlatMap(
    Pure(999_999),
    x => FlatMap(
      ...
        Pure(0)
      ...
    )
  )
)
```

No computation has happened. The `sum(n-1)` inside the lambda is not called
until the runtime evaluates that lambda. Building the tree only ever recurses
one level deep per `sum` call — each call constructs one `FlatMap` node and
returns immediately.

The cats-effect `IORuntime` then walks the tree iteratively — a trampoline loop
— never growing the stack.

---

## IO as a description, not an execution

This is the deeper consequence: `IO[A]` is a **value** that describes a
computation. It is not the computation itself.

```scala
val program: IO[Unit] =
  IO.println("hello") *> IO.println("world")

// Nothing has printed yet. program is just data.

program.unsafeRunSync()   // NOW the runtime interprets the tree and prints
```

You can:
- Pass `program` to a function
- Run it twice
- Add a timeout: `program.timeout(5.seconds)`
- Retry on failure: `program.retry(...)`
- Never run it at all

This is only possible because `IO` is a data structure, not a side effect.
The trampoline is what makes that data structure safe to build recursively.

---

## Summary

| Concept | What it means |
|---|---|
| Tail recursion | Recursive call is last — frame can be reused |
| Trampoline | Return `More(thunk)` instead of recursing — pending work goes on heap |
| `run()` | The only recursive function; always tail-recursive |
| `IO` | A sealed ADT — program as data; runtime interprets it iteratively |
| `flatMap` | Builds a `FlatMap` node on the heap — does NOT call the next step yet |
| `unsafeRunSync` | The trampoline loop — walks the IO tree and executes it |
