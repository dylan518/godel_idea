"""Runner: execute an idea generator system on benchmark topics and save results.

Each run writes a run_config.json alongside ideas.json. Before using cached
results, godel_loop.py calls cache_is_valid() to check model/n_ideas match.

Usage:
    python ideas/runner.py --system S0 --output ideas/results/S0/
    python ideas/runner.py --system S0 --output ideas/results/S0/ --n-ideas 3
"""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

_log.load_dotenv()
logger = _log.setup("runner")


def load_system(version: str, systems_dir: str):
    """Load a generator system by version string (e.g. 'S0')."""
    module_path = os.path.join(systems_dir, f"{version}.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"System file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(version, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "GENERATOR"):
        raise AttributeError(f"{version}.py must define a module-level GENERATOR singleton")

    return module.GENERATOR


def load_topics(topics_path: str) -> list[dict]:
    with open(topics_path) as f:
        data = json.load(f)
    return data["topics"]


def cache_is_valid(output_dir: str, model: str, n_ideas: int, n_topics: int | None = None) -> bool:
    """Return True if output_dir has a valid ideas.json matching current model/n_ideas/n_topics."""
    ideas_path = Path(output_dir) / "ideas.json"
    config_path = Path(output_dir) / "run_config.json"
    if not ideas_path.exists() or not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        ok = cfg.get("model") == model and cfg.get("n_ideas") == n_ideas
        if ok and n_topics is not None:
            ok = cfg.get("n_topics") == n_topics
        return ok
    except Exception:
        return False


def _flush_partial(results: list[dict], out: Path):
    """Atomically write partial results so crashes don't lose completed work."""
    tmp = out / "ideas.partial.json"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(out / "ideas.json")


def run_system(
    version: str,
    topics: list[dict],
    output_dir: str,
    model: str,
    n_ideas: int,
    systems_dir: str,
    workers: int = 1,
    fresh: bool = False,
) -> list[dict]:
    """Run a generator system on all topics and save ideas to output_dir/ideas.json.

    Always starts clean: clears output_dir before writing.
    Saves incrementally as work completes so crashes don't lose completed work.
    workers > 1 runs all independent work concurrently. Systems that override
    generate_batch() get one task per topic; otherwise each topic×idea pair is
    scheduled independently.

    fresh=True: use generate_batch() — one LLM call per topic returns all n_ideas
    at once.  Results are NOT cached (run_config.json omitted) so the next call
    always regenerates.  Used by full-eval comparisons to prevent overfitting to
    a fixed idea set.
    """
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import IdeaGenerator, make_client

    generator = load_system(version, systems_dir)
    client = make_client(model)

    # Clean workspace: wipe and recreate output dir
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = len(topics) * n_ideas
    results = []
    flush_lock = Lock()
    topic_order = {t["id"]: i for i, t in enumerate(topics)}

    has_custom_batch = type(generator).generate_batch is not IdeaGenerator.generate_batch
    use_batch = fresh or (n_ideas > 1 and has_custom_batch)

    def _run_topic_fresh(topic_entry: dict) -> list[dict]:
        """Batch mode: one LLM call produces all n_ideas for this topic."""
        topic_id = topic_entry["id"]
        topic = topic_entry["topic"]
        logger.info("[%s] %s (batch×%d)", topic_id, topic, n_ideas)
        try:
            batch = generator.generate_batch(topic, client, model=model, n=n_ideas)
            logger.info("  [%s] batch ok — %d ideas", topic_id, len(batch))
        except Exception as e:
            logger.error("  [%s] batch FAILED [%s]: %s", topic_id, type(e).__name__, e)
            batch = [f"ERROR: {e}"] * n_ideas
        return [
            {
                "topic_id": topic_id,
                "topic": topic,
                "domain": topic_entry.get("domain", ""),
                "idea_index": i,
                "text": batch[i] if i < len(batch) else "ERROR: missing",
                "system_version": generator.VERSION,
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(n_ideas)
        ]

    def _run_idea(topic_entry: dict, idea_idx: int) -> dict:
        topic_id = topic_entry["id"]
        topic = topic_entry["topic"]
        logger.info("[%s] %s (idea %d/%d)", topic_id, topic, idea_idx + 1, n_ideas)
        try:
            text = generator.generate_idea(topic, client, model=model)
            logger.info("  [%s] idea %d/%d — ok (%d chars)",
                        topic_id, idea_idx + 1, n_ideas, len(text))
        except TimeoutError as e:
            logger.error("  [%s] idea %d/%d — TIMEOUT: %s", topic_id, idea_idx + 1, n_ideas, e)
            text = f"TIMEOUT: {e}"
        except Exception as e:
            logger.error("  [%s] idea %d/%d — FAILED [%s]: %s",
                         topic_id, idea_idx + 1, n_ideas, type(e).__name__, e)
            text = f"ERROR: {e}"
        return {
            "topic_id": topic_id,
            "topic": topic,
            "domain": topic_entry.get("domain", ""),
            "idea_index": idea_idx,
            "text": text,
            "system_version": generator.VERSION,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Flush on SIGTERM/SIGINT so partial results survive a kill
    # Signal handlers can only be set in the main thread; skip when called from workers
    import threading
    if threading.current_thread() is threading.main_thread():
        def _on_signal(sig, frame):
            with flush_lock:
                if results:
                    _flush_partial(results, out)
                    logger.warning("Interrupted (signal %d) — saved %d partial results to %s",
                                   sig, len(results), out / "ideas.json")
            sys.exit(1)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

    if use_batch:
        work_units = len(topics)
        effective_workers = min(max(1, workers), work_units) if work_units else 1
        mode_label = "batch" if not fresh else "fresh-batch"
    else:
        work_units = len(topics) * n_ideas
        effective_workers = min(max(1, workers), work_units) if work_units else 1
        mode_label = "idea-parallel"

    # Generators that call the shared Swiss tournament can use otherwise-idle
    # runner capacity inside each topic task.
    prev_tournament_workers = os.environ.get("IDEAS_TOURNAMENT_WORKERS")
    nested_workers = max(1, workers // max(1, effective_workers))
    if prev_tournament_workers is None:
        os.environ["IDEAS_TOURNAMENT_WORKERS"] = str(nested_workers)

    logger.info("Running %s on %d topic(s) × %d idea(s) = %d calls  "
                "[%s, workers=%d]",
                version, len(topics), n_ideas,
                len(topics) if use_batch else total,
                mode_label, effective_workers)

    try:
        if effective_workers <= 1:
            if use_batch:
                for topic_entry in topics:
                    topic_results = _run_topic_fresh(topic_entry)
                    results.extend(topic_results)
                    results.sort(key=lambda r: (topic_order.get(r["topic_id"], 10**9), r["idea_index"]))
                    _flush_partial(results, out)
                    logger.debug("  flushed %d/%d ideas to disk", len(results), total)
            else:
                for topic_entry in topics:
                    for idea_idx in range(n_ideas):
                        results.append(_run_idea(topic_entry, idea_idx))
                        results.sort(key=lambda r: (topic_order.get(r["topic_id"], 10**9), r["idea_index"]))
                        _flush_partial(results, out)
                        logger.debug("  flushed %d/%d ideas to disk", len(results), total)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                if use_batch:
                    future_map = {executor.submit(_run_topic_fresh, t): t for t in topics}
                else:
                    future_map = {
                        executor.submit(_run_idea, t, i): (t, i)
                        for t in topics for i in range(n_ideas)
                    }
                for future in as_completed(future_map):
                    completed = future.result()
                    with flush_lock:
                        if isinstance(completed, list):
                            results.extend(completed)
                        else:
                            results.append(completed)
                        results.sort(key=lambda r: (topic_order.get(r["topic_id"], 10**9), r["idea_index"]))
                        _flush_partial(results, out)
                        logger.debug("  flushed %d/%d ideas to disk", len(results), total)
    finally:
        if prev_tournament_workers is None:
            os.environ.pop("IDEAS_TOURNAMENT_WORKERS", None)

    # Fresh mode: skip run_config.json so cache_is_valid() returns False next time,
    # forcing regeneration on the next call.
    if fresh:
        logger.info("Saved %d ideas to %s (fresh — not cached)", len(results), out / "ideas.json")
        return results

    config_path = out / "run_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "system_version": version,
            "model": model,
            "n_ideas": n_ideas,
            "n_topics": len(topics),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    logger.info("Saved %d ideas to %s", len(results), out / "ideas.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run an idea generator system on benchmark topics")
    parser.add_argument("--system", required=True, help="System version to run, e.g. S0")
    parser.add_argument("--output", required=True, help="Output directory for ideas.json")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="Model to use (default: gpt-4.1-mini)")
    parser.add_argument("--n-ideas", type=int, default=1,
                        help="Number of ideas per topic (default: 1)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel topic workers (default: 1; use e.g. 15 for full benchmark speed)",
    )
    parser.add_argument("--topics", default=None,
                        help="Path to benchmark_topics.json (default: auto-detected)")
    args = parser.parse_args()

    ideas_dir = Path(__file__).parent
    systems_dir = str(ideas_dir / "systems")
    topics_path = args.topics or str(ideas_dir / "benchmark_topics.json")

    logger.info("system=%s  model=%s  n_ideas=%d", args.system, args.model, args.n_ideas)
    topics = load_topics(topics_path)
    logger.info("Loaded %d benchmark topics", len(topics))

    run_system(
        version=args.system,
        topics=topics,
        output_dir=args.output,
        model=args.model,
        n_ideas=args.n_ideas,
        systems_dir=systems_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
