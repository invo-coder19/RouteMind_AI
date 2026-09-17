# Prompt Complexity Labeling Rubric

## Purpose
This rubric defines the criteria used to assign a complexity label (`simple`,
`moderate`, or `complex`) to every prompt in `labeled_prompts.csv`.  Follow
these rules exactly when adding new prompts so the dataset remains consistent.

---

## Tier Definitions

### `simple`
**Characteristics**
- Single-fact retrieval or definition look-up
- Short, direct-answer question (answer fits in 1–3 sentences)
- Basic reformatting or rewriting with no analysis required
- No reasoning chain needed; the answer is a known fact or obvious inference

**Signals to look for**
- Starts with *what is*, *who is*, *when did*, *define*, *list*
- Answer can be found verbatim in a textbook or encyclopedia
- Task: translate a word, fix a typo, convert a unit

**Examples**

| Prompt | Why `simple` |
|--------|-------------|
| "What is the capital of France?" | Single-fact retrieval, no reasoning |
| "Define the word 'ephemeral'." | Dictionary look-up |
| "Convert 100 degrees Fahrenheit to Celsius." | One-step arithmetic formula |

---

### `moderate`
**Characteristics**
- Requires some structure or explanation — not just a single fact
- Single-step reasoning or comparison between two things
- Summarization of a provided text
- Explanation that benefits from an example but doesn't need multi-step planning

**Signals to look for**
- Starts with *how does*, *explain*, *compare*, *summarize*, *what are the differences between*
- Answer typically requires 1–3 paragraphs
- Might involve selecting among options but not planning a multi-step solution

**Examples**

| Prompt | Why `moderate` |
|--------|---------------|
| "Explain how HTTPS works." | Needs structured explanation, not just one fact |
| "Compare SQL and NoSQL databases." | Single-step comparison |
| "Summarize the following paragraph: …" | Summarization requires comprehension |

---

### `complex`
**Characteristics**
- Multi-step reasoning — the model must plan before answering
- Code generation, debugging, or architectural design
- Open-ended or ambiguous tasks requiring judgment calls
- Tasks that require integrating multiple concepts or sources

**Signals to look for**
- Starts with *write a program*, *debug this*, *design a system*, *how would you approach*
- Answer involves several distinct steps or sub-tasks
- Requires trade-off analysis, creative generation, or domain expertise
- Ambiguous intent that requires the model to ask clarifying questions or make assumptions explicit

**Examples**

| Prompt | Why `complex` |
|--------|--------------|
| "Write a Python function to parse nested JSON and flatten it to a single-level dict." | Code generation with edge-case reasoning |
| "Design a microservices architecture for a ride-sharing app. Justify your choices." | Multi-step design + justification |
| "Why might two economists disagree about the long-term effects of QE?" | Open-ended reasoning requiring integration of competing theories |

---

## Boundary Guidelines

- If a prompt sits on the `simple`/`moderate` boundary, ask: *does answering require more than one sentence of structured reasoning?*  If yes → `moderate`.
- If a prompt sits on the `moderate`/`complex` boundary, ask: *does the model need to plan or iterate before producing the answer?*  If yes → `complex`.
- Prompts with multi-part questions (e.g. "What is X and how does it differ from Y and what are the pros/cons?") are at least `moderate`, likely `complex`.
