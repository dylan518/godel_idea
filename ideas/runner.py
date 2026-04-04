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
) -> list[dict]:
    """Run a generator system on all topics and save ideas to output_dir/ideas.json.

    Always starts clean: clears output_dir before writing.
    Saves incrementally after each topic so crashes don't lose completed work.
    workers > 1 runs topics concurrently (ideas within each topic stay sequential
    to avoid burst rate-limiting).
    """
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client

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

    def _run_topic(topic_entry: dict) -> list[dict]:
        topic_id = topic_entry["id"]
        topic = topic_entry["topic"]
        logger.info("[%s] %s", topic_id, topic)
        topic_results = []
        for idea_idx in range(n_ideas):
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
            topic_results.append({
                "topic_id": topic_id,
                "topic": topic,
                "domain": topic_entry.get("domain", ""),
                "idea_index": idea_idx,
                "text": text,
                "system_version": generator.VERSION,
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return topic_results

    # Flush on SIGTERM/SIGINT so partial results survive a kill
    def _on_signal(sig, frame):
        with flush_lock:
            if results:
                _flush_partial(results, out)
                logger.warning("Interrupted (signal %d) — saved %d partial results to %s",
                               sig, len(results), out / "ideas.json")
        sys.exit(1)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    effective_workers = min(workers, len(topics))
    logger.info("Running %s on %d topic(s) × %d idea(s) = %d calls  "
                "[clean workspace, workers=%d]",
                version, len(topics), n_ideas, total, effective_workers)

    if effective_workers <= 1:
        for topic_entry in topics:
            topic_results = _run_topic(topic_entry)
            results.extend(topic_results)
            _flush_partial(results, out)
            logger.debug("  flushed %d/%d ideas to disk", len(results), total)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {executor.submit(_run_topic, t): t for t in topics}
            for future in as_completed(future_map):
                topic_results = future.result()
                with flush_lock:
                    results.extend(topic_results)
                    _flush_partial(results, out)
                    logger.debug("  flushed %d/%d ideas to disk", len(results), total)

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
    )


if __name__ == "__main__":
    main()
