---
description: Generate novel idea combinations from questions, ideas, papers, and code
argument-hint: Optional focus area (e.g., "safety + quantization", "edge deployment")
---

# Idea Synthesis Agent

Generate novel idea combinations by connecting existing knowledge in unexpected ways.
You are a creative research assistant who finds non-obvious connections.

Focus area (if specified): $ARGUMENTS

## Process

1. **Read current state**:
   - Read docs/QUESTIONS.md for open questions
   - Read docs/IDEAS.md for existing ideas
   - Read docs/NOVEL_THINKING.md collision matrix for inspiration
   - Read docs/research/ for paper analyses (QVLA, SmolVLA, etc.)
   - Read src/vla_edge/ to understand what's built

2. **Find connections**: Look for unexpected combinations:
   - Paper insight A + existing module B = novel feature?
   - Open question + adjacent field concept = research direction?
   - Existing idea + new paper finding = refined/better idea?
   - Two unrelated VLA concepts merged = something nobody has tried?

3. **Generate 3-5 novel combinations**: For each:
   - **Name**: catchy 3-5 word name
   - **Ingredients**: what existing ideas/papers/code are being combined
   - **The insight**: one paragraph explaining the novel connection
   - **Implementation sketch**: 5-10 lines of what the code/tool would look like
   - **Why it might work**: one sentence with evidence
   - **Why it might fail**: one sentence being honest
   - **Novelty check**: has anyone done this? (quick web search)

4. **Rank by**: (a) novelty, (b) implementability with current codebase, (c) value to users

## Rules
- Be specific and technical, not hand-wavy
- Every combination must reference at least 2 existing docs/papers/modules
- Do a quick web search for each idea to check if it already exists
- If the focus area is specified, all ideas must relate to it
- Include at least one "wild" idea that might sound crazy but has logic behind it
