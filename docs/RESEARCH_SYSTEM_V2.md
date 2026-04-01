# Research Agent System v2: Multi-Agent Discovery Engine

**Inspired by**: Virtual Biotech (CSO + 37K specialist agents), NOVEL_THINKING.md
(collision matrix, bisociation, first principles, Hamming questions)

---

## Architecture: 5 Specialists + 1 Custodian + 1 Arbiter

### The Custodian (always-on context keeper)

Like the Virtual Biotech's CSO, the Custodian maintains the full picture.
It reads ALL agent outputs and maintains a structured knowledge state.

**What it knows at all times:**
- Current codebase state (what's built, what's missing)
- All research findings (papers, benchmarks, competitor moves)
- Open questions (QUESTIONS.md)
- Active ideas (IDEAS.md)
- Collision matrix state (which cells explored, which unexplored)
- What each specialist agent has produced
- Contradictions between specialist findings

**What it does:**
- After each specialist completes, updates the shared knowledge base
- Detects when two specialists' findings create a novel combination
- Flags contradictions for the Arbiter to resolve
- Maintains a running "state of knowledge" summary

### 5 Specialist Agents

| Agent | Focus | Novel Thinking Method | Knowledge Base |
|-------|-------|----------------------|----------------|
| **Efficiency** | Compression, quantization, pruning | First Principles (what's the minimum compute?) | QVLA, DyQ-VLA, ProbeFlow, NanoVLA, BitVLA |
| **Systems** | Runtime, pipelining, caching, hardware | Bisociation (what solved this in other fields?) | ActionFlow, VLASH, VLA-Perf, DuoCore-FS |
| **Safety** | Robustness, verification, contracts | TRIZ Contradictions (accuracy vs speed) | SafeVLA, RobustVLA, @safety_contract |
| **Bio-inspired** | Virtual Biotech patterns, multi-agent coordination | Adjacent Possible (one step from what exists) | Virtual Biotech, lora-router, adapter routing |
| **Evaluation** | Benchmarks, metrics, statistical rigor | Hamming Questions (what's important?) | LIBERO, CALVIN, vla-eval, MedAgentBench |

### The Arbiter (synthesis + decision)

Reads all specialist outputs + Custodian's knowledge state.
Applies: collision matrix (combine ideas from different specialists),
Hamming audit (is this an important problem?), feasibility filter.

---

## Discussion Protocol (runs as background agents)

```
Phase 1: DIVERGE (parallel, background)
  - Launch all 5 specialists simultaneously
  - Each proposes 2-3 ideas from their domain
  - Each applies their assigned novel thinking method
  - Custodian collects and indexes all outputs

Phase 2: CROSS-EXAMINE (parallel, background)
  - Each specialist reads the OTHER specialists' proposals
  - Each writes critiques from their perspective
  - Safety agent: "does this optimization break safety?"
  - Efficiency agent: "is this systems trick actually slower?"
  - Evaluation agent: "can we measure this on LIBERO?"

Phase 3: SYNTHESIZE (Arbiter, foreground)
  - Reads all proposals + all critiques + Custodian summary
  - Applies collision matrix: combine ideas across domains
  - Applies Hamming: is this an important problem?
  - Outputs: Top 3 ranked ideas with full experimental plan
```

---

## Novel Thinking Integration

Each specialist uses a DIFFERENT thinking method (from NOVEL_THINKING.md):

**Efficiency Agent - First Principles:**
"Strip VLA inference to physics: what's the minimum FLOPs for a 7-DoF
action prediction? If the theoretical minimum is X and we're at 100X,
where are the 99X wasted?"

**Systems Agent - Bisociation:**
"What solved latency problems in audio DSP? Game engines? Mobile ML?
What if we applied [other field's solution] to VLA inference?"

**Safety Agent - TRIZ Contradictions:**
"We want more accuracy AND more speed. These contradict. TRIZ says:
resolve by separating in time (high accuracy for grasps, low for transit)
or space (high accuracy for end-effector, low for base)."

**Bio-inspired Agent - Adjacent Possible:**
"Virtual Biotech routes to specialist agents based on query type.
What's one step away? Route to specialist VLA adapters based on
task phase. The adapter registry already exists in vla-edge."

**Evaluation Agent - Hamming Questions:**
"What are the top 3 problems in VLA edge deployment RIGHT NOW?
Is what we're proposing one of them? If not, why are we working on it?"

---

## Background Execution Model

All research runs as background agents. Your screen can lock.

```
You run: /research "optimize SmolVLA for Jetson"
  |
  +-> Background: Efficiency agent researches (5-10 min)
  +-> Background: Systems agent researches (5-10 min)
  +-> Background: Safety agent researches (5-10 min)
  +-> Background: Bio-inspired agent researches (5-10 min)
  +-> Background: Evaluation agent researches (5-10 min)
  |
  [All complete, Custodian collects]
  |
  +-> Background: Cross-examination round (5 min each, parallel)
  |
  [All complete]
  |
  You review: Arbiter synthesis with top 3 ideas + experimental plans
```

Total: ~20-30 min of background agent time, 5 min of your review time.

---

## Custodian's Knowledge State (structured JSON)

```json
{
  "last_updated": "2026-03-31",
  "codebase": {
    "models": ["smolvla", "openvla"],
    "backends": ["cpu", "cuda", "mps", "jetson"],
    "optimizations": ["probeflow"],
    "safety": ["safety_contract", "safety_guard"],
    "benchmarks": {"smolvla_cpu": {"cold_ms": 28165, "probeflow_ms": 11935}}
  },
  "research_frontier": {
    "most_promising": "safety-aware adaptive denoising",
    "open_questions": 17,
    "unexplored_collisions": 20,
    "papers_tracked": 35
  },
  "contradictions": [
    "ProbeFlow saves time but L1=0.15 divergence may hurt task success",
    "GGUF is proven on Jetson but QVLA shows VLA-specific quant is better"
  ]
}
```

---

## Implementing as a Skill: /research

```markdown
---
description: Launch multi-agent research system for novel VLA optimization ideas
argument-hint: Research topic (e.g., "optimize SmolVLA for Jetson")
---

Launch 5 specialist background agents + Custodian to research and debate.
Results saved to docs/research/sessions/. Arbiter synthesis presented
when all agents complete.
```

---

## Paper Target

The system itself isn't the paper. The OUTPUTS are the paper:

**"Safety-Aware Adaptive Flow Matching for Edge-Deployed VLA Models"**

Novel: Connect safety metrics to denoising step allocation.
When robot is near objects -> more steps (safer, slower).
When in open transit -> fewer steps (faster, acceptable risk).

Measurable on LIBERO with safety + success + latency metrics.
Nobody has done this (verified by Safety agent search).
