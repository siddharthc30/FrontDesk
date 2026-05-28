# Hotel NL Search — Eval Suite

This eval suite detects three categories of silent failures in the hotel NL search pipeline:

| Risk | What can go wrong |
|------|-------------------|
| **Routing accuracy** | The LLM router sends a query to the wrong path (e.g. simple filter → expensive fallback SQL) |
| **Query correctness** | The generated SQL/spec doesn't match user intent (e.g. `MAX` instead of `AVG`, wrong city filter) |
| **Narration faithfulness** | The answer invents hotels, statistics, or claims not present in the returned rows |

---

## Quick start

### 1. Set environment variables

Add these to your `.env` (same file used to run the app):

```
# Langfuse — required for evals
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# LLM provider — same as running the app (e.g. OpenAI or Gemini)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Sign up free at https://cloud.langfuse.com to get Langfuse keys.

### 2. Run the eval suite

```bash
python -m evals.run_eval
```

The script will:
1. Upsert the 22 test items into a Langfuse dataset called `hotel-nl-search-eval`.
2. Run every item through the pipeline.
3. Attach deterministic scores to each trace in Langfuse.
4. Call the LLM judge (~15–18 calls) to score narration faithfulness.
5. Print a summary table to the console.
6. Print a direct URL to the Langfuse experiment run.

---

## Viewing results in Langfuse

1. Go to your Langfuse project → **Datasets** → **hotel-nl-search-eval**.
2. Click on the experiment run (named `eval-run-<timestamp>`).
3. Each row shows one test question with all its scores side-by-side.
4. To compare two runs after a prompt change, select both runs in the sidebar.

---

## Score definitions

| Score | Type | Meaning |
|-------|------|---------|
| `routing_correct` | BOOLEAN | Did the router pick the expected path? |
| `grounding_min_hotels` | BOOLEAN | Did the result contain at least the expected number of hotels? |
| `grounding_max_hotels` | BOOLEAN | Did the result contain at most the expected number of hotels? |
| `grounding_all_city` | BOOLEAN | Do all returned hotels match the expected city? |
| `grounding_all_min_score` | BOOLEAN | Do all returned hotels meet the minimum score threshold? |
| `grounding_all_has_amenity` | BOOLEAN | Do all returned hotels have the required amenity flag set? |
| `grounding_answer_contains` | BOOLEAN | Does the answer text contain the expected substrings? |
| `grounding_should_decline` | BOOLEAN | Did declined results actually decline (and vice versa)? |
| `has_query_ran` | BOOLEAN | Was the executed query surfaced in `query_ran`? (transparency check) |
| `empty_result_honest` | BOOLEAN | When 0 hotels returned, does the answer admit it (vs. inventing data)? |
| `narration_faithful` | NUMERIC 0–1 | LLM judge: is the answer grounded only in the returned rows? |

---

## Adding new test cases

Open [evals/dataset.py](dataset.py) and append an `EvalItem` to `EVAL_DATASET`:

```python
EvalItem(
    id="my_test_01",
    question="Hotels in Vienna with a sauna",
    expected_path="parameterized",
    assertions={
        "min_hotels": 1,
        "all_city": "Vienna",
        "all_has_amenity": "sauna",
    },
    notes="City + amenity filter for Vienna.",
),
```

Then re-run `python -m evals.run_eval`. The new item is automatically upserted to Langfuse.

### Assertion reference

```python
assertions = {
    # Result counts
    "min_hotels": 1,           # at least N hotels
    "max_hotels": 5,           # at most N hotels
    "exact_hotels": 3,         # exactly N hotels

    # Column checks (on every returned hotel)
    "all_city": "Paris",       # city must match (case-insensitive)
    "all_min_score": 8.5,      # avg_score >= this value
    "all_has_amenity": "pool", # has_pool == 1 (also: wifi, gym, sauna, restaurant, ...)

    # Answer text checks
    "answer_contains": ["Paris", "pool"],    # all substrings must appear
    "answer_not_contains": ["Tokyo"],        # none of these must appear

    # Decline check
    "should_decline": True,    # expects result.declined == True
}
```

---

## Architecture

```
evals/
  dataset.py      # 22 test cases as EvalItem dataclasses
  evaluators.py   # score_routing, score_result_grounding, score_transparency,
                  # score_empty_result_honesty, score_narration_faithfulness
  run_eval.py     # orchestrates dataset upsert + experiment run + summary
  README.md       # this file
```

The evaluators import `core.models.PipelineResponse` and `core.llm.chat_completion` but
do **not** modify any pipeline code. The eval suite is purely additive.
