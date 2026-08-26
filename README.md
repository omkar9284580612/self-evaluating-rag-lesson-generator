# Self-Evaluating Lesson Content Generator

An agentic system that generates a beginner-friendly lesson on a given
topic, evaluates its output against a hard pass/fail rubric, and
regenerates it using targeted feedback when the draft is rejected.

The system uses LangGraph to orchestrate the generate → evaluate →
retry workflow and persistent JSON memory to learn from repeated
failure patterns across runs.

Built for the NxtWave GenAI Engineer — Content Systems take-home.

## Why this architecture

### LangGraph orchestration

The workflow is represented as an explicit graph:

generate → evaluate → (ship OR prepare retry → generate)

A conditional edge makes the quality decision, while a hard retry
counter guarantees termination.

LangGraph makes the agentic control flow visible and easy to extend
with additional validation, tools, or human review later.

### Binary pass/fail rubric

The evaluator does not assign a vague score such as "7/10".

Instead, every quality checkpoint is either:

- PASS
- FAIL

When a checkpoint fails, the evaluator provides a specific reason.

For example:

```text
FAIL: uses the term "embedding" without defining it.
This feedback is directly passed to the generator so the next draft knows what needs to be fixed.

The six checkpoints are defined in:

src/rubric.py
Provider-agnostic LLM client

src/llm_client.py contains the provider-specific API logic.

The rest of the application communicates through:

LLMClient.complete(...)

The architecture supports:

Google Gemini
Anthropic Claude
OpenAI

This means the generator and evaluator do not need to know which provider is being used.

A provider and model can be selected through environment variables.

Two-tier persistent memory

The system maintains two kinds of memory.

1. Per-run audit logs

Stored under:

memory/runs/

Each run records:

topic
final status
every generation attempt
evaluator results
failure reasons
generated lesson snapshots
final lesson

This provides an audit trail showing how the lesson changed across attempts.

2. Cross-run lessons learned

Stored in:

memory/lessons_learned.json

Every failed rubric checkpoint is counted globally.

When a particular failure pattern occurs repeatedly, it is promoted into a standing rule.

That rule is then injected into future generation prompts.

For example:

no_unexplained_jargon

may become a standing instruction such as:

Past runs repeatedly failed on 'no_unexplained_jargon':
technical terms were introduced without simple definitions.

This allows the system to improve across different runs without a human manually editing the generation prompt after every failure.

## Guaranteed termination

The current configuration uses:

MAX_RETRIES = 2

This means:

Attempt 1 → Initial generation

Attempt 2 → First retry using evaluator feedback

Attempt 3 → Second retry using evaluator feedback

Therefore, the pipeline performs a maximum of 3 generation attempts per run.

If any attempt passes all rubric checkpoints, the lesson is shipped immediately.

If the third attempt still fails, the pipeline terminates with:

MAX_RETRIES_REACHED

The best-effort final lesson and complete rejection log are still written to the outputs/ directory.

This hard limit prevents the agent from entering an infinite generate/evaluate loop.

Project structure
lesson-agent/
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── rubric.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── memory_store.py
│   └── orchestrator.py
├── memory/
│   ├── runs/
│   └── lessons_learned.json
└── outputs/
    ├── lesson_final.md
    └── rejection_log.md
Architecture flow
                         ┌──────────────┐
                         │    START     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   GENERATE   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   EVALUATE   │
                         └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   PASS                    FAIL
                    │                       │
                    ▼                       ▼
                 ┌──────┐          ┌────────────────┐
                 │ SHIP │          │ Retry available│
                 └──┬───┘          └───────┬────────┘
                    │                      │
                    │                      ▼
                    │               ┌──────────────┐
                    │               │ PREPARE RETRY│
                    │               └──────┬───────┘
                    │                      │
                    │                      ▼
                    │               ┌──────────────┐
                    │               │   GENERATE   │
                    │               └──────┬───────┘
                    │                      │
                    │                      ▼
                    │               ┌──────────────┐
                    │               │   EVALUATE   │
                    │               └──────┬───────┘
                    │                      │
                    │                PASS / FAIL
                    │                      │
                    ▼                      ▼
                   END              SHIP / MAX_RETRIES

The evaluator uses six hard PASS/FAIL checkpoints.

Failed reasons are passed back to the generator.

Persistent failure patterns are counted in:

memory/lessons_learned.json

Repeated failure patterns become standing rules for future generation runs.

Setup
1. Get an API key

Choose one supported provider.

For Gemini, create an API key through Google AI Studio.

The API key should be stored only in .env.

Do not put the real API key in:

.env.example
README.md
GitHub
source code

Example .env:

LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=your_api_key_here

The .env file is excluded from Git through .gitignore.

2. Clone and install
git clone <your-repo-url>
cd lesson-agent
pip install -r requirements.txt

Create .env from .env.example and add your API key.

3. Run the pipeline
python main.py --topic "RAG (Retrieval-Augmented Generation)"
4. Check the outputs

After execution, the following files are created:

outputs/lesson_final.md
outputs/rejection_log.md

lesson_final.md contains the final generated lesson.

rejection_log.md contains the evaluator's results for each attempt.

Demo mode

The project includes a deterministic demo mode for showing the self-evaluation and retry behavior.

Run:

python main.py --topic "RAG (Retrieval-Augmented Generation)" --inject-error

With --inject-error, the first generated lesson is intentionally given two problems:

It incorrectly claims that RAG works by fine-tuning the LLM's internal weights.
It uses the term cosine similarity without defining it.

The evaluator should detect these problems and return FAIL results.

The failed reasons are then passed to the generator.

The next generation is produced without deliberate error injection and uses the evaluator's feedback.

If the second attempt passes all checkpoints, it is shipped.

If it still fails, the pipeline terminates with:

MAX_RETRIES_REACHED
Output files
outputs/lesson_final.md

Contains the final lesson produced by the pipeline.

outputs/rejection_log.md

Contains:

each attempt
every rubric checkpoint
PASS/FAIL status
evaluator reasoning
feedback passed to the next generation
memory/runs/

Contains JSON audit logs for individual pipeline runs.

memory/lessons_learned.json

Contains:

failure counts
graduated standing rules
Design trade-offs
Single evaluator call

The evaluator currently evaluates all six checkpoints in one LLM call.

A more robust production implementation could use:

one evaluator call per checkpoint, or
a second independent model as a cross-check.

This would reduce the risk of the evaluator incorrectly judging the generator's output, but would increase latency and API cost.

JSON file memory

Persistent memory is implemented using JSON files.

This is intentionally simple and sufficient for the scope of this project.

At production scale, the memory system could be moved to a database or vector store.

A semantic retrieval system could also retrieve only the most relevant past lessons for the current topic instead of injecting all standing rules.

No RAG inside the generator

The project itself is about building a self-evaluating lesson generation system.

The generator does not use a RAG pipeline to create the lesson.

For a future version, trusted reference documents could be retrieved before generation and supplied to the model as context.

The evaluator could then additionally verify whether generated claims are supported by those references.

Limitations

The system depends on the availability and rate limits of the selected LLM provider.

A provider may return:

rate-limit errors
temporary service errors
network timeouts
incomplete generations

The LLM client retries temporary API failures according to its retry configuration.

The pipeline also has a hard generation-attempt limit so that provider or model problems cannot create an infinite loop.

Future improvements

Possible extensions include:

deterministic lesson completeness validation
stronger output-length handling
independent evaluator model
reference-document retrieval
citation checking
semantic long-term memory
database-backed run history
human approval before shipping
automatic provider fallback when one provider is unavailable
