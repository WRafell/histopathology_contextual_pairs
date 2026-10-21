# Claude–Codex Collaboration Skill

## Purpose

Use Claude Code and Codex as complementary coding agents rather than redundant ones.

- **Active agent**: whichever agent the human is currently asking—Codex or Claude Code—acts as the primary agent for that task.
- **Peer agent**: the other agent acts as an independent reviewer, critic, alternative hypothesis generator, or second implementation voice when useful.
- **Human**: retains decision authority when agents disagree.

There is no fixed primary agent. Primary responsibility follows the user's entry point for each task.

The goal is to improve reliability through independent analysis and critique without creating recursive agent loops.

---

## Activation and Trigger Rules

This collaboration protocol does **not** need to be explicitly requested every time.

The active agent should decide whether to invoke the peer agent based on task value and expected benefit.

### Invoke collaboration automatically when

Use the peer agent when one or more of the following applies:

- the user explicitly asks for collaboration, council, second opinion, critique, review, or independent analysis;
- the task has multiple plausible technical approaches;
- the design is architecturally significant;
- debugging is difficult, ambiguous, or repeated attempts have failed;
- the change touches persistent state, synchronization, concurrency, distributed behavior, or optimizer logic;
- scientific or experimental validity is involved;
- the task could materially affect conclusions, benchmarks, or research claims;
- the implementation is substantial enough to benefit from independent final review;
- there is meaningful uncertainty that can be reduced through an independent perspective.

### Do not invoke collaboration automatically when

Avoid peer consultation for low-value or deterministic tasks such as:

- trivial edits;
- formatting;
- renaming;
- obvious syntax fixes;
- simple file lookup;
- straightforward shell commands;
- deterministic mechanical transformations;
- questions that can be answered confidently without repository inspection;
- tiny changes that can be verified immediately.

### Explicit user override

The user always has final control.

If the user says:

```text
Do not ask the other agent.
```

or equivalent, do not invoke collaboration.

If the user says:

```text
Use both agents.
```

or asks for a council, second opinion, independent review, or cross-check, collaboration is required.

### Default behavior

When uncertain, prefer the following rule:

```text
If a second independent analysis is likely to change the implementation,
catch a meaningful error, or reveal a hidden assumption, consult the peer.
Otherwise, proceed alone.
```

The objective is not to maximize the number of agent calls. The objective is to use the second agent only when it improves decision quality.

### Activation hierarchy

Use this order of priority:

```text
1. Explicit user instruction
2. High-risk / high-uncertainty task
3. Research-validity or architecture-sensitive task
4. Substantial implementation ready for review
5. Otherwise, single-agent execution
```

---

## Available Commands

Consult Claude from Codex:

```bash
claude -p "<question>"
```

Consult Codex from Claude:

```bash
codex exec "<question>"
```

Prefer non-interactive calls for agent-to-agent communication.

---

## Core Protocol

For non-trivial tasks:

1. Analyze the task independently.
2. Form an initial hypothesis or implementation plan.
3. Decide whether a second opinion is useful.
4. Ask the other agent a **focused, self-contained question**.
5. Compare its answer against your own reasoning.
6. Resolve disagreements using repository evidence, tests, documentation, or experiments.
7. Implement the selected approach.
8. Run relevant tests/checks.
9. For substantial changes, request an independent final review.
10. Fix only review comments that are supported by evidence.

Do not treat the other agent as an authority.

---

## Role Assignment

Primary responsibility is dynamic.

- If the human asks **Codex**, Codex is primary and Claude is the peer reviewer.
- If the human asks **Claude**, Claude is primary and Codex is the peer reviewer.
- Either agent may implement, debug, refactor, run tests, or design architecture.
- The peer agent should normally provide an independent opinion rather than taking over the whole task.

### When Codex Is Active

Codex may use Claude for:

- independent architecture proposals
- alternative debugging hypotheses
- methodology critique
- edge-case discovery
- final diff review
- a second implementation strategy when the design is ambiguous

### When Claude Is Active

Claude may use Codex for:

- independent architecture proposals
- repository-grounded debugging
- concrete implementation alternatives
- test strategy
- edge-case discovery
- final diff review

The active agent remains responsible for integrating the evidence and completing the task.

## Consultation Rules

A consultation prompt should include enough context to be useful without requiring the second agent to reconstruct the whole conversation.

Good:

```bash
claude -p "
We are modifying a PyTorch federated-learning codebase.

Goal:
Synchronize Muon momentum every K rounds while model weights synchronize every round.

Current behavior:
- model weights: FedAvg every round
- momentum: local only

Question:
What is the cleanest state-management design for adding momentum synchronization
without coupling it to weight aggregation?

Focus on architecture and failure modes. Do not edit files.
"
```

Bad:

```bash
claude -p "what do you think?"
```

Keep consultations narrow.

---

## Independent-Opinion Rule

When requesting a second opinion, avoid revealing your preferred conclusion unless necessary.

Preferred:

```text
Independently diagnose the likely cause of this failure.
List evidence that would distinguish competing explanations.
```

Avoid:

```text
I think the dataloader is broken. Confirm that this is the problem.
```

This reduces anchoring between agents.

---

## Disagreement Protocol

If Claude and Codex disagree:

1. State the disagreement explicitly.
2. Identify the assumptions behind each view.
3. Find evidence available in:
   - source code
   - tests
   - logs
   - documentation
   - reproducible experiments
4. Run the smallest decisive test.
5. Prefer empirical evidence over agent confidence.
6. If uncertainty remains, present the alternatives to the human rather than hiding the disagreement.

Do not synthesize two incompatible answers into a fake consensus.

---

## Final Review Protocol

For substantial changes, the implementation agent should ask the other agent to review the actual diff.

Example from Codex:

```bash
git diff > /tmp/current.diff

claude -p "
Review the current repository changes as a senior engineer.

Focus on:
- correctness
- regressions
- hidden assumptions
- state-management bugs
- missing tests
- unnecessary complexity

Inspect the git diff and repository as needed.

Do not rewrite the implementation.
Return only actionable findings, ordered by severity.
"
```

If direct repository access is available to the reviewing agent, prefer that over pasting a large diff into the prompt.

---

## Research-Code Protocol

For scientific or ML code, review should additionally check:

- train/validation/test leakage
- seed handling
- unfair baseline differences
- inconsistent preprocessing
- accidental use of future/global information
- optimizer-state reset behavior
- incorrect aggregation weights
- evaluation-mode mistakes
- metric implementation
- logging correctness
- checkpoint comparability
- hidden changes in hyperparameters

For experimental claims, distinguish:

```text
code works
```

from:

```text
experiment supports the hypothesis
```

These are separate questions.

---

## Anti-Recursion Rule

Never allow uncontrolled delegation loops.

If Codex calls Claude:

```text
Claude must not call Codex during this consultation.
```

If Claude calls Codex:

```text
Codex must not call Claude during this consultation.
```

A useful prompt suffix is:

```text
Answer independently. Do not delegate this task to another coding agent.
```

Maximum recommended depth:

```text
Human
  -> Primary agent
      -> Reviewer agent
```

Do not create:

```text
Codex -> Claude -> Codex -> Claude -> ...
```

---

## Cost and Context Discipline

Do not consult the second agent for every trivial operation.

No consultation is normally needed for:

- renaming a variable
- formatting
- obvious syntax fixes
- simple file lookup
- deterministic mechanical edits
- commands whose result can be verified immediately

Consult when the expected value of an independent opinion exceeds the added latency and token cost.

Good triggers:

- multiple plausible implementations
- unclear root cause
- broad refactor
- concurrency/state issues
- experiment methodology
- destructive change
- repeated failed attempts
- security-sensitive behavior
- surprising test results

---

## Suggested Instruction When Codex Is Active

Use this when Codex is the primary agent:

```text
You are the active primary agent for this task because the user invoked you.

You may consult Claude Code using:

    claude -p "<question>"

Use Claude only when an independent second opinion is likely to improve the result,
especially for architecture, difficult debugging, methodology, or final review.

Protocol:
1. Analyze independently first.
2. Form your own hypothesis.
3. Ask Claude a focused question without unnecessarily revealing your preferred answer.
4. Compare its response with your own analysis.
5. Resolve disagreements using code, tests, logs, documentation, or experiments.
6. Implement the best-supported approach.
7. Run verification.
8. For substantial changes, ask Claude to review the final diff.
9. Apply review comments only when technically justified.

Claude is a reviewer, not an authority.

Never create recursive delegation.
When consulting Claude, explicitly tell it not to call Codex.
```

---

## Suggested Instruction When Claude Is Active

Use this when Claude is the primary agent:

```text
You are the active primary agent for this task because the user invoked you.

You may consult Codex using:

    codex exec "<question>"

Use Codex for independent repository-grounded analysis, alternate implementations,
debugging hypotheses, test strategy, and final code review.

Protocol:
1. Analyze independently first.
2. Form your own hypothesis.
3. Ask Codex a focused, self-contained question.
4. Avoid leading Codex toward your preferred answer.
5. Compare the two analyses.
6. Resolve disagreements using repository evidence and executable tests.
7. Implement only the best-supported approach.
8. For substantial changes, request a final independent review.

Codex is a reviewer, not an authority.

Never create recursive delegation.
When consulting Codex, explicitly tell it not to call Claude.
```

---

## Symmetry Principle

The collaboration is peer-to-peer:

```text
User -> Codex
        Codex = primary
        Claude = peer

User -> Claude
        Claude = primary
        Codex = peer
```

Do not hard-code one model as the permanent implementer or permanent reviewer.

The primary agent may ask the peer to review, critique, propose an alternative, or independently implement a candidate solution. The primary agent still owns the final integration for that task unless the human explicitly reassigns responsibility.

---

## Council Mode

For difficult design or research questions, use both agents independently before synthesis.

Workflow:

```text
Question
   |-------------------|
   v                   v
Codex               Claude
independent         independent
analysis            analysis
   |                   |
   |---------+---------|
             v
        comparison
             |
             v
      evidence/test
             |
             v
       final decision
```

Recommended sequence:

1. Ask Codex independently.
2. Ask Claude independently.
3. Compare:
   - agreements
   - disagreements
   - assumptions
   - missing evidence
4. Run a decisive experiment when possible.
5. Produce a final synthesis that preserves unresolved uncertainty.

---

## Example: Debugging

```bash
CODEX_VIEW=$(codex exec "
Independently diagnose why validation loss becomes NaN after round 20.
Inspect the repository if needed.
Return likely causes, supporting evidence, and decisive tests.
Do not call Claude.
")

CLAUDE_VIEW=$(claude -p "
Independently diagnose why validation loss becomes NaN after round 20.
Inspect the repository if needed.
Return likely causes, supporting evidence, and decisive tests.
Do not call Codex.
")
```

Then compare the two reports rather than asking one agent to summarize the other blindly.

---

## Example: Architecture Decision

Primary agent prompt:

```text
We need to add optimizer-memory synchronization at a timescale independent from
model-weight synchronization.

Before editing:

1. Propose your own architecture.
2. Ask the second agent for an independent architecture.
3. Compare both designs in terms of:
   - state ownership
   - synchronization timing
   - partial participation
   - checkpointing
   - backward compatibility
   - testability
4. Choose based on concrete repository constraints.
5. Implement.
6. Run tests.
7. Request final independent review.
```

---

## Example: Final Diff Review

```bash
claude -p "
Act as an independent code reviewer.

Review the current uncommitted changes.

Check:
1. correctness
2. regressions
3. edge cases
4. API compatibility
5. unnecessary complexity
6. missing tests
7. experiment-validity issues

Do not call Codex.
Do not modify files.
Return actionable findings only.
"
```

---

## Operating Principle

The collaboration should behave like:

```text
independent reasoning
        +
adversarial review
        +
empirical verification
        =
higher-confidence implementation
```

Not like:

```text
one model generates an answer
        ->
the second model agrees
        ->
confidence increases without evidence
```

The purpose of the second agent is to create useful disagreement, expose blind spots,
and suggest tests—not merely to endorse the first agent.
