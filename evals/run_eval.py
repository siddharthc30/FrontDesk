"""Langfuse evaluation pipeline runner for the hotel NL search system.

Runs every item in EVAL_DATASET through the pipeline, attaches deterministic
and LLM-as-judge scores to each trace in Langfuse, and prints a summary table.

Usage:
    python -m evals.run_eval

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the environment (or .env).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# ── Env-var guard ─────────────────────────────────────────────────────────────

_REQUIRED_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
if _missing:
    print(
        f"ERROR: Missing required environment variables: {', '.join(_missing)}\n"
        "Set them in your .env file or environment before running evals.\n"
        "Example:\n"
        "  LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
        "  LANGFUSE_SECRET_KEY=sk-lf-...\n"
        "  LANGFUSE_BASE_URL=https://cloud.langfuse.com"
    )
    sys.exit(1)

# ── Langfuse init ─────────────────────────────────────────────────────────────

from langfuse import Evaluation, get_client  # noqa: E402

langfuse = get_client()

# ── Dataset + imports ─────────────────────────────────────────────────────────

from evals.dataset import EVAL_DATASET  # noqa: E402
from evals.evaluators import (  # noqa: E402
    score_empty_result_honesty,
    score_narration_faithfulness,
    score_result_grounding,
    score_routing,
    score_transparency,
)

DATASET_NAME = "hotel-nl-search-eval"


# ── Step 1: Upsert dataset items in Langfuse ──────────────────────────────────

def upsert_dataset() -> None:
    """Create the Langfuse dataset (idempotent) and upsert all eval items."""
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Evaluation dataset for the hotel NL search pipeline",
    )

    for item in EVAL_DATASET:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item.id,
            input={
                "question": item.question,
                "user_lat": item.user_lat,
                "user_lng": item.user_lng,
            },
            expected_output={
                "expected_path": item.expected_path,
                "assertions": item.assertions or {},
                "notes": item.notes,
            },
        )

    print(f"✓ Upserted {len(EVAL_DATASET)} items to Langfuse dataset '{DATASET_NAME}'")


# ── Step 2: Task function ─────────────────────────────────────────────────────

async def _async_eval_task(*, item, **kwargs):
    """Call the pipeline and return a PipelineResponse."""
    from core.pipeline import ask

    input_data = item.input if hasattr(item, "input") else item["input"]

    try:
        result = await ask(
            question=input_data.get("question", "") or "",
            user_lat=input_data.get("user_lat"),
            user_lng=input_data.get("user_lng"),
        )
    except Exception as exc:  # noqa: BLE001
        from core.models import PipelineResponse
        result = PipelineResponse(
            answer=f"Pipeline crashed: {exc}",
            insights="",
            hotels=[],
            path="declined",
            declined=True,
            decline_reason=str(exc),
        )

    return result



# ── Step 3: Evaluator function ────────────────────────────────────────────────

async def evaluate_item(*, input, output, expected_output, metadata=None, **kwargs):
    """Item-level evaluator: deterministic checks + LLM faithfulness judge."""
    scores: list[Evaluation] = []

    # Guard: output may be None if the task raised unexpectedly
    if output is None:
        from core.models import PipelineResponse
        output = PipelineResponse(
            answer="",
            insights="",
            hotels=[],
            path="declined",
            declined=True,
            decline_reason="task returned None",
        )

    expected_path = expected_output.get("expected_path", "")
    assertions = expected_output.get("assertions", {})

    # 1. Routing
    scores.append(score_routing(output, expected_path))

    # 2. Result grounding
    if assertions:
        scores.extend(score_result_grounding(output, assertions))

    # 3. Empty result honesty
    honesty = score_empty_result_honesty(output)
    if honesty is not None:
        scores.append(honesty)

    # 4. Transparency
    scores.append(score_transparency(output))

    # 5. LLM faithfulness judge (parameterized + fallback only)
    # Skipped for review_search: returned HotelRow objects don't contain review text,
    # so the judge can't verify claims about guest mentions — it would always fail.
    if not output.declined and output.hotels and output.path not in ("review_search",):
        question = (input or {}).get("question", "")
        faithfulness = await score_narration_faithfulness(output, question)
        scores.append(faithfulness)

    return scores


# ── Step 4: Summary printer ───────────────────────────────────────────────────

def _print_summary(experiment_result) -> None:
    item_results = experiment_result.item_results or []

    total = len(item_results)
    if total == 0:
        print("\nNo results to summarise.")
        return

    counters: dict[str, list[bool | float]] = {}
    failures: list[str] = []

    for ir in item_results:
        item_id = "?"
        try:
            item_id = (ir.item.id if hasattr(ir.item, "id") else
                       ir.item.get("id", "?") if isinstance(ir.item, dict) else "?")
        except Exception:  # noqa: BLE001
            pass

        evaluations = ir.evaluations or []
        for ev in evaluations:
            name = ev.name
            val = ev.value
            counters.setdefault(name, []).append(val)

            if isinstance(val, bool) and not val:
                failures.append(f"  {item_id}: {name}=False  [{ev.comment or ''}]")
            elif isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0.7:
                failures.append(
                    f"  {item_id}: {name}={val:.2f}  [{ev.comment or ''}]"
                )

    print("\n=== Hotel NL Search Eval Results ===")
    print(f"Total items:  {total}")
    print()

    for name, values in sorted(counters.items()):
        booleans = [v for v in values if isinstance(v, bool)]
        numerics = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]

        if booleans:
            n_true = sum(1 for v in booleans if v)
            n_total = len(booleans)
            pct = 100 * n_true / n_total if n_total else 0
            print(f"  {name}: {n_true}/{n_total} ({pct:.1f}%)")

        if numerics:
            avg = sum(numerics) / len(numerics)
            print(f"  {name}: avg={avg:.3f} over {len(numerics)} items")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
    else:
        print("\nAll checks passed.")

    # Print Langfuse URL if available
    url = getattr(experiment_result, "dataset_run_url", None)
    if url:
        print(f"\nLangfuse results: {url}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Hotel NL Search Eval — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"Dataset: {DATASET_NAME}  |  Items: {len(EVAL_DATASET)}\n")

    # 1. Upsert dataset items in Langfuse
    upsert_dataset()

    # 2. Fetch items back so the experiment is linked to the Langfuse dataset
    dataset = langfuse.get_dataset(DATASET_NAME)

    # 3. Run experiment
    run_name = f"eval-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    print(f"\nRunning experiment '{run_name}' …")

    experiment_result = langfuse.run_experiment(
        name=DATASET_NAME,
        run_name=run_name,
        description="Automated eval: routing accuracy, result grounding, narration faithfulness",
        data=dataset.items,
        task=_async_eval_task,
        evaluators=[evaluate_item],
        max_concurrency=5,  # conservative — LLM calls inside
    )

    # 4. Print summary
    _print_summary(experiment_result)

    # 5. Flush remaining Langfuse spans before exit
    langfuse.flush()


if __name__ == "__main__":
    main()
