---
description: Launch multi-agent research system for novel VLA optimization ideas
argument-hint: Research topic (e.g., "optimize SmolVLA denoising", "safety-aware scheduling")
---

# Multi-Agent Research System

Launch 5 specialist agents in parallel (background) to research, debate,
and propose novel optimizations. Based on docs/RESEARCH_SYSTEM_V2.md.

Topic: $ARGUMENTS

## Phase 1: Launch Specialists (all background)

Launch these 5 agents simultaneously as background tasks:

### Agent 1: Efficiency Researcher
"You are an ML efficiency researcher. Topic: [TOPIC]. Apply FIRST PRINCIPLES
thinking - what's the theoretical minimum compute for this operation?
Propose 2-3 novel optimizations. Each must be training-free, measurable on LIBERO,
and novel (verify via web search). Include: name, mechanism, expected speedup, risks.
Knowledge: QVLA, DyQ-VLA, ProbeFlow, NanoVLA, BitVLA, SQAP-VLA."

### Agent 2: Systems Researcher
"You are a systems researcher. Topic: [TOPIC]. Apply BISOCIATION thinking -
what solved this in game engines? Audio DSP? Mobile ML? DevOps?
Propose 2-3 system-level optimizations. Each training-free, measurable on LIBERO.
Knowledge: ActionFlow, VLASH, DuoCore-FS, VLA-Perf, FASTER, AsyncVLA."

### Agent 3: Safety Researcher
"You are a safety researcher. Topic: [TOPIC]. Apply TRIZ CONTRADICTION thinking -
the user wants X AND Y which contradict. How to resolve without compromise?
Propose 1-2 safety-aware optimization ideas. Validate that proposed optimizations
don't degrade safety. Knowledge: SafeVLA, RobustVLA, @safety_contract, ASIMOV."

### Agent 4: Bio-inspired / Multi-Agent Researcher
"You are a multi-agent systems researcher. Topic: [TOPIC]. Apply ADJACENT POSSIBLE
thinking - what's one step away from what vla-edge already has?
Propose 1-2 ideas using multi-agent patterns (routing, specialist delegation,
hierarchical planning). Knowledge: Virtual Biotech, lora-router, SP-VLA."

### Agent 5: Evaluation Researcher
"You are an evaluation researcher. Topic: [TOPIC]. Apply HAMMING QUESTIONS -
is this an important problem? What are the top 3 problems in this area?
Design the evaluation plan: datasets, metrics, baselines, ablations, episode counts.
Knowledge: LIBERO, CALVIN, vla-eval, MetaWorld."

## Phase 2: Custodian Collects

After all 5 complete, read all outputs. Build a unified summary:
- All proposals (with source agent)
- Contradictions between proposals
- Overlapping ideas that could combine

## Phase 3: Arbiter Synthesis

Apply collision matrix: combine ideas from different agents.
Apply Hamming: is each idea addressing an important problem?
Rank top 3 by: novelty x feasibility x measurability x paper potential.

For each top idea:
- One-paragraph description
- Which specialist agents contributed
- Experimental plan (dataset, metrics, baselines)
- Implementation estimate (hours/days)
- Paper venue recommendation

Save results to docs/research/sessions/[DATE]_[TOPIC].md
