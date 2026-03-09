---
name: physics-tdd
description: Bottom-up TDD workflow for physics engine — build correctness from primitives up through composed functions
disable-model-invocation: true
---

# Physics TDD Workflow

You are implementing and testing a physics engine where correctness is critical. Every function must be proven correct before anything that depends on it is built.

Follow this bottom-up process strictly.

## Phase 1: Dependency Analysis

1. Read all source files under `engine/physics/` and all test files under `engine/tests/`.
2. Build a dependency graph of functions — which functions call which.
3. Identify the **leaf functions** (functions that depend on no other project functions, only on constants/math).
4. Rank all functions into layers:
   - **Layer 0**: Leaf functions (no internal dependencies)
   - **Layer 1**: Functions that only depend on Layer 0
   - **Layer 2**: Functions that only depend on Layers 0-1
   - ...and so on
5. Present the full dependency graph and layer assignments to the user.
6. For each function, report its current status:
   - **Untested**: No tests exist
   - **Partially tested**: Some tests exist but coverage of base cases, edge cases, or self-consistency checks is incomplete
   - **Fully tested**: Base cases, edge cases, and self-consistency checks all covered
7. **Stop and wait for user approval** before proceeding.

## Phase 2: Implement and Test Layer by Layer

Starting from Layer 0, work through each layer. Within each layer, work through one function at a time.

### For each function:

#### Step A: Write Tests First

Write tests covering three categories:

1. **Base cases** — Simple, straightforward inputs where the expected result is easy to verify by hand or by physical intuition.
   - Example: A ball struck straight along the x-axis with known speed.

2. **Edge cases** — Boundary conditions and degenerate inputs.
   - Zero velocity, zero spin, ball already at the target position
   - Events that should never happen (e.g. time-to-stop for an already-stopped ball should return `None`)
   - Negative or extreme values where applicable

3. **Self-consistency checks** — Use the function's own outputs to verify invariants.
   - If you compute `time_to_stop`, then evaluate the motion at that time — velocity must be zero.
   - If you compute `time_to_event`, then evaluate state at that time — the event condition must hold.
   - Conservation laws where applicable (energy, momentum).
   - Symmetry: mirrored inputs should give mirrored outputs.

For numerical assertions, use `Decimal` with explicit precision (`THREE_PLACES = Decimal('0.000')`) to make expected values readable and verifiable.

Compute expected values by hand or with an independent calculation (inline Python in comments is fine) — never just "run it and paste the output".

#### Step B: Report and Await Approval

Present to the user:
- The function signature and a one-line description of what it does
- The full list of test cases you will add, grouped by category (base / edge / self-consistency)
- For base cases, show the hand calculation or reasoning behind expected values

**Stop and wait for user approval before writing any implementation code.**

#### Step C: Run Tests (expect failures for new functions)

Run the new tests. For functions being newly implemented, confirm they fail as expected. For functions being newly tested (already implemented), note any failures — these indicate bugs.

#### Step D: Implement or Fix

- If the function is new: write the minimal implementation to pass all tests.
- If the function exists but tests revealed bugs: fix the bugs.
- If all tests pass already: move on.

#### Step E: Verify and Report

Run the full test suite (`python -m pytest engine/tests/ -v`). Report:
- All tests pass / which tests fail
- A one-line summary: "Function `X` — fully tested and passing"

**Stop and wait for user approval before moving to the next function.**

### After completing a layer:

Run the full test suite and report a layer summary:
- List every function in the layer with its status
- Confirm no regressions in earlier layers

**Stop and wait for user approval before moving to the next layer.**

## Guidelines

- Never skip the approval steps. The user must confirm at every gate.
- Keep tests focused — one concept per test function.
- Test names should read as documentation: `test_time_rolling_to_stop_returns_none_when_stationary`.
- Do not modify existing passing tests unless they are wrong (test the wrong physics).
- When a test uses a composed calculation (e.g. slide then roll), earlier layers must already be proven correct.
- If you discover that a lower-layer function has a bug while testing a higher layer, stop, go fix and re-verify the lower layer first.
