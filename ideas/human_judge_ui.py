#!/usr/bin/env python3
"""Local web UI for blind human pairwise scoring of generated ideas.

Pick domains you know well, choose two result runs, then **Enter blind mode**:
generator names are hidden, the two picks are randomly mapped to API
left/right so picker order does not reveal columns, and each pair shuffles
Idea A/B. ``/api/next`` accepts **POST JSON** (preferred; avoids URL-length limits on long
custom topics) or **GET** query params. It does not return version ids — only the page merges them
when recording a vote. Scores append to ``results/human_blind_scores.jsonl``.

With **custom topic** text, ``/api/next`` **auto-generates** matching ideas for
both runs (cached in-memory) if needed. If a **benchmark topic** is selected
but cached ``ideas.json`` files do not overlap, the server generates fresh ideas
for that benchmark title instead of erroring.

Usage (from repo root)::

    python3 ideas/human_judge_ui.py
    python3 ideas/human_judge_ui.py --port 8765 --no-browser
    python3 ideas/human_judge_ui.py --model gpt-4.1   # server default for generation

Static UI is stdlib-only; generation uses the same Python deps as ``runner.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

IDEAS_DIR = Path(__file__).resolve().parent
if str(IDEAS_DIR) not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR))
import log as _log

_log.load_dotenv()
RESULTS_DIR = IDEAS_DIR / "results"
STATIC_DIR = IDEAS_DIR / "human_judge_static"
SCORES_LOG = RESULTS_DIR / "human_blind_scores.jsonl"
BENCHMARK_TOPICS = IDEAS_DIR / "benchmark_topics.json"
DEV_TOPICS = IDEAS_DIR / "dev_topics.json"
CUSTOM_PREFIX = "CUSTOM"
DEFAULT_CUSTOM_N_IDEAS = 5

from godel_loop import DEFAULT_MODEL as DEFAULT_GENERATION_MODEL

CUSTOM_IDEA_CACHE: dict[tuple[str, str, str, str, int, str], tuple[list[dict], list[dict]]] = {}


def _load_benchmark_topics() -> list[dict]:
    topics: list[dict] = []
    for path in (BENCHMARK_TOPICS, DEV_TOPICS):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        topics.extend(data.get("topics", []))
    return topics


def _topic_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in _load_benchmark_topics():
        tid = t.get("id")
        if tid:
            out[tid] = t
    return out


def _discover_versions() -> list[str]:
    if not RESULTS_DIR.is_dir():
        return []
    versions = []
    for p in sorted(RESULTS_DIR.iterdir()):
        if p.is_dir() and (p / "ideas.json").is_file():
            versions.append(p.name)
    return versions


def _load_ideas(version: str) -> list[dict]:
    path = RESULTS_DIR / version / "ideas.json"
    if not path.is_file():
        raise FileNotFoundError(f"No ideas file for {version}: {path}")
    return json.loads(path.read_text())


def _index_ideas(rows: list[dict]) -> dict[tuple[str, int], dict]:
    idx: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["topic_id"], row["idea_index"])
        idx[key] = row
    return idx


def _common_pairs(
    left_rows: list[dict],
    right_rows: list[dict],
    domain: str | None,
    topic_id: str | None,
    topic_contains: str | None = None,
) -> list[tuple[str, int]]:
    li = _index_ideas(left_rows)
    ri = _index_ideas(right_rows)
    keys = sorted(set(li.keys()) & set(ri.keys()))
    needle = topic_contains.strip().lower() if topic_contains else None
    out: list[tuple[str, int]] = []
    for k in keys:
        a, b = li[k], ri[k]
        if domain and (a.get("domain") != domain or b.get("domain") != domain):
            continue
        if topic_id:
            if k[0] != topic_id:
                continue
        elif needle:
            title = (a.get("topic") or "").lower()
            if needle not in title:
                continue
        out.append(k)
    return out


def _resolve_winner(
    winner: str,
    swap: bool,
    left_version: str,
    right_version: str,
) -> tuple[str | None, str | None]:
    """Return (winner_version or None for tie, loser_version or None)."""
    if winner == "tie":
        return None, None
    if not swap:
        if winner == "A":
            return left_version, right_version
        return right_version, left_version
    if winner == "A":
        return right_version, left_version
    return left_version, right_version


def _json_body_to_qs(body: object) -> dict[str, list[str]]:
    """Convert a flat JSON object into ``parse_qs``-style single-string lists."""
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    out: dict[str, list[str]] = {}
    for k, v in body.items():
        key = str(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[key] = [str(x) for x in v]
        else:
            out[key] = [str(v)]
    return out


def _custom_cache_key(
    left: str,
    right: str,
    topic: str,
    domain: str | None,
    n_ideas: int,
    model: str,
) -> tuple[str, str, str, str, int, str]:
    return (
        left.strip(),
        right.strip(),
        topic.strip().lower(),
        (domain or "").strip().lower(),
        n_ideas,
        model.strip(),
    )


def _load_generator(version: str):
    from runner import load_system

    return load_system(version, str(IDEAS_DIR / "systems"))


def _make_client(model: str):
    systems_dir = str(IDEAS_DIR / "systems")
    if systems_dir not in sys.path:
        sys.path.insert(0, systems_dir)
    from base import make_client

    return make_client(model)


def _ensure_custom_cached(
    left: str,
    right: str,
    topic: str,
    domain: str | None,
    n_ideas: int,
    model: str,
) -> tuple[list[dict], list[dict]]:
    """Populate CUSTOM_IDEA_CACHE for this triple if missing; return both row lists."""
    key = _custom_cache_key(left, right, topic, domain, n_ideas, model)
    if key not in CUSTOM_IDEA_CACHE:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fl = pool.submit(_generate_custom_rows, left, topic, domain, n_ideas, model)
            fr = pool.submit(_generate_custom_rows, right, topic, domain, n_ideas, model)
            CUSTOM_IDEA_CACHE[key] = (fl.result(), fr.result())
    return CUSTOM_IDEA_CACHE[key]


def _generate_custom_rows(
    version: str,
    topic: str,
    domain: str | None,
    n_ideas: int,
    model: str,
) -> list[dict]:
    generator = _load_generator(version)
    client = _make_client(model)
    try:
        ideas = generator.generate_batch(topic, client, model=model, n=n_ideas)
    except Exception:
        ideas = [
            generator.generate_idea(topic, client, model=model)
            for _ in range(n_ideas)
        ]
    topic_id = f"{CUSTOM_PREFIX}:{topic.strip().lower()}"
    return [
        {
            "topic_id": topic_id,
            "topic": topic,
            "domain": domain or "",
            "idea_index": i,
            "text": ideas[i] if i < len(ideas) else "ERROR: missing generated idea",
            "system_version": version,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for i in range(n_ideas)
    ]


def _pair_context_from_qs(
    qs: dict[str, list[str]],
    default_model: str,
) -> tuple[list[dict], list[dict], list[tuple[str, int]]]:
    """Load rows and the sorted list of (topic_id, idea_index) pairs shared by both runs."""
    left = (qs.get("left") or [""])[0].strip()
    right = (qs.get("right") or [""])[0].strip()
    domain = (qs.get("domain") or [""])[0].strip() or None
    custom_topic = (qs.get("custom_topic") or [""])[0].strip() or None
    try:
        custom_n = int((qs.get("custom_n") or [str(DEFAULT_CUSTOM_N_IDEAS)])[0])
    except ValueError:
        custom_n = DEFAULT_CUSTOM_N_IDEAS
    custom_n = max(1, min(12, custom_n))
    topic_id = (qs.get("topic_id") or [""])[0].strip() or None
    topic_contains = (qs.get("topic_contains") or [""])[0].strip() or None
    if custom_topic:
        topic_id = None
        topic_contains = None
    if topic_contains:
        topic_id = None
    gen_model = (qs.get("model") or [""])[0].strip() or default_model
    if not left or not right:
        raise ValueError("Select two versions")
    if left == right:
        raise ValueError("Pick two different versions")
    if custom_topic:
        left_rows, right_rows = _ensure_custom_cached(
            left, right, custom_topic, domain, custom_n, gen_model
        )
        raw_pairs = _common_pairs(left_rows, right_rows, domain, None, None)
    else:
        left_rows = _load_ideas(left)
        right_rows = _load_ideas(right)
        raw_pairs = _common_pairs(
            left_rows, right_rows, domain, topic_id, topic_contains
        )
        if not raw_pairs and topic_id:
            tinfo = _topic_by_id().get(topic_id)
            if tinfo:
                tt = str(tinfo.get("topic") or "").strip()
                dom = (tinfo.get("domain") or "").strip() or None
                if tt:
                    left_rows, right_rows = _ensure_custom_cached(
                        left,
                        right,
                        tt,
                        dom or domain,
                        custom_n,
                        gen_model,
                    )
                    raw_pairs = _common_pairs(
                        left_rows, right_rows, domain, None, None
                    )
        if not raw_pairs and topic_contains:
            left_rows, right_rows = _ensure_custom_cached(
                left,
                right,
                topic_contains,
                domain,
                custom_n,
                gen_model,
            )
            raw_pairs = _common_pairs(left_rows, right_rows, domain, None, None)
    if not raw_pairs:
        raise ValueError(
            "No overlapping topic/idea_index rows for this filter, and nothing "
            "could be auto-generated. Enter a custom topic (full research question), "
            "pick a benchmark topic from the dropdown, or relax domain/topic filters."
        )
    pairs = sorted(set(raw_pairs))
    return left_rows, right_rows, pairs


def _try_fast_custom_deck(qs: dict[str, list[str]]) -> dict | None:
    """If the client is using a custom topic string, return slot metadata without LLM calls.

    Slot ids match ``_generate_custom_rows`` so ``POST /api/next`` can fill the cache lazily.
    """
    custom_topic = (qs.get("custom_topic") or [""])[0].strip() or None
    if not custom_topic:
        return None
    left = (qs.get("left") or [""])[0].strip()
    right = (qs.get("right") or [""])[0].strip()
    if not left or not right:
        raise ValueError("Select two versions")
    if left == right:
        raise ValueError("Pick two different versions")
    domain = (qs.get("domain") or [""])[0].strip() or None
    try:
        custom_n = int((qs.get("custom_n") or [str(DEFAULT_CUSTOM_N_IDEAS)])[0])
    except ValueError:
        custom_n = DEFAULT_CUSTOM_N_IDEAS
    custom_n = max(1, min(12, custom_n))
    tid = f"{CUSTOM_PREFIX}:{custom_topic.strip().lower()}"
    slots = [
        {
            "topic_id": tid,
            "idea_index": i,
            "topic": custom_topic.strip(),
            "domain": domain or "",
        }
        for i in range(custom_n)
    ]
    return {"slot_count": len(slots), "slots": slots}


class Handler(SimpleHTTPRequestHandler):
    """generation_model is set in main() from CLI (default = godel_loop idea model)."""

    generation_model: str = DEFAULT_GENERATION_MODEL

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self):
        p = getattr(self, "path", "") or ""
        if isinstance(p, str):
            base = p.split("?", 1)[0]
            if base.endswith((".html", ".js", ".css")):
                self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api"):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()
            return
        self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            try:
                self._send_json(self._config_payload())
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": f"server error: {e!s}"}, status=500)
            return
        if parsed.path == "/api/next":
            qs = parse_qs(parsed.query)
            try:
                payload = self._next_pair(qs)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": f"server error: {e!s}"}, status=500)
                return
            self._send_json(payload)
            return
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/deck":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, status=400)
                return
            try:
                qs = _json_body_to_qs(body)
                deck = self._deck_payload(qs)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": f"server error: {e!s}"}, status=500)
                return
            self._send_json(deck)
            return
        if parsed.path == "/api/next":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, status=400)
                return
            try:
                qs = _json_body_to_qs(body)
                payload = self._next_pair(qs)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": f"server error: {e!s}"}, status=500)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/generate_custom":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, status=400)
                return
            try:
                result = self._generate_custom(body)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": f"server error: {e!s}"}, status=500)
                return
            self._send_json(result)
            return
        if parsed.path != "/api/vote":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return
        try:
            reveal = self._record_vote(body)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": f"server error: {e!s}"}, status=500)
            return
        self._send_json(reveal)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _config_payload(self) -> dict:
        topics = _load_benchmark_topics()
        domains = sorted({t.get("domain", "") for t in topics if t.get("domain")})
        return {
            "versions": _discover_versions(),
            "domains": domains,
            "default_model": type(self).generation_model,
            "topics": [
                {"id": t["id"], "topic": t["topic"], "domain": t.get("domain", "")}
                for t in topics
                if t.get("id")
            ],
        }

    def _deck_payload(self, qs: dict[str, list[str]]) -> dict:
        fast = _try_fast_custom_deck(qs)
        if fast is not None:
            return fast
        left_rows, right_rows, pairs = _pair_context_from_qs(qs, type(self).generation_model)
        li = _index_ideas(left_rows)
        slots: list[dict] = []
        for topic_id_i, idea_index in pairs:
            row = li[(topic_id_i, idea_index)]
            slots.append(
                {
                    "topic_id": topic_id_i,
                    "idea_index": idea_index,
                    "topic": row.get("topic", ""),
                    "domain": row.get("domain", ""),
                }
            )
        return {"slot_count": len(slots), "slots": slots}

    def _next_pair(self, qs: dict[str, list[str]]) -> dict:
        left_rows, right_rows, pairs = _pair_context_from_qs(qs, type(self).generation_model)
        pick_tid = (qs.get("pick_topic_id") or [""])[0].strip() or None
        raw_ii = (qs.get("pick_idea_index") or [""])[0].strip()
        if not pick_tid or raw_ii == "":
            raise ValueError(
                "Missing pick_topic_id and pick_idea_index. POST /api/deck with the same "
                "filters first, then POST /api/next once per entry in slots (each pair once)."
            )
        try:
            pick_ii = int(raw_ii)
        except ValueError as e:
            raise ValueError("pick_idea_index must be an integer") from e
        key = (pick_tid, pick_ii)
        if key not in set(pairs):
            raise ValueError(
                "That slot is not in the current deck — rebuild the deck or fix pick fields."
            )
        topic_id_i, idea_index = key
        li = _index_ideas(left_rows)
        ri = _index_ideas(right_rows)
        row_l = li[(topic_id_i, idea_index)]
        row_r = ri[(topic_id_i, idea_index)]
        swap = random.random() < 0.5
        if not swap:
            text_a, text_b = row_l["text"], row_r["text"]
        else:
            text_a, text_b = row_r["text"], row_l["text"]
        # Do not include generator ids here — the UI keeps them client-side only so the
        # browser never receives which run is "left" until after a vote is recorded.
        return {
            "swap": swap,
            "topic_id": topic_id_i,
            "idea_index": idea_index,
            "pool_size": len(pairs),
            "topic": row_l.get("topic", ""),
            "domain": row_l.get("domain", ""),
            "idea_a": text_a,
            "idea_b": text_b,
        }

    def _generate_custom(self, body: dict) -> dict:
        left = str(body.get("left", "")).strip()
        right = str(body.get("right", "")).strip()
        topic = str(body.get("topic", "")).strip()
        domain = str(body.get("domain", "")).strip() or None
        model = str(body.get("model", Handler.generation_model)).strip() or Handler.generation_model
        n_ideas = int(body.get("n_ideas", DEFAULT_CUSTOM_N_IDEAS))
        if not left or not right:
            raise ValueError("Select two versions")
        if left == right:
            raise ValueError("Pick two different versions")
        if not topic:
            raise ValueError("Enter a custom topic first")
        if n_ideas < 1 or n_ideas > 12:
            raise ValueError("n_ideas must be between 1 and 12")
        print(
            f"[human_judge_ui] generate_custom start: {left} vs {right} model={model} n_ideas={n_ideas}",
            file=sys.stderr,
            flush=True,
        )
        _ensure_custom_cached(left, right, topic, domain, n_ideas, model)
        print("[human_judge_ui] generate_custom done.", file=sys.stderr, flush=True)
        return {"ok": True, "cached": True, "n_ideas": n_ideas, "topic": topic}

    def _record_vote(self, body: dict) -> dict:
        required = [
            "left_version",
            "right_version",
            "swap",
            "topic_id",
            "idea_index",
            "winner",
        ]
        for k in required:
            if k not in body:
                raise ValueError(f"Missing field: {k}")
        winner = body["winner"]
        if winner not in ("A", "B", "tie"):
            raise ValueError("winner must be A, B, or tie")
        swap = bool(body["swap"])
        left_v = str(body["left_version"])
        right_v = str(body["right_version"])
        w_ver, l_ver = _resolve_winner(winner, swap, left_v, right_v)
        idea_a_ver = right_v if swap else left_v
        idea_b_ver = left_v if swap else right_v
        scorer = str(body.get("scorer", "")).strip()
        if not scorer:
            raise ValueError(
                "Missing rater name. Enter your name in the 'Scorer' field "
                "before submitting a vote. Required for inter-rater analysis."
            )
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "scorer": scorer,
            "rater_id": scorer,
            "topic_id": body["topic_id"],
            "idea_index": body["idea_index"],
            "topic": body.get("topic", ""),
            "domain": body.get("domain", ""),
            "left_version": left_v,
            "right_version": right_v,
            "swap": swap,
            "winner_label": winner,
            "winner_version": w_ver,
            "loser_version": l_ver,
            "scores_a": body.get("scores_a"),
            "scores_b": body.get("scores_b"),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCORES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "ok": True,
            "idea_a_version": idea_a_ver,
            "idea_b_version": idea_b_ver,
            "winner_version": w_ver,
            "loser_version": l_ver,
        }

    def _send_json(self, obj: dict, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)


class _DualStackAllInterfaces(HTTPServer):
    """Listen on ``::`` with ``IPV6_V6ONLY=0`` so IPv4 and IPv6 clients can connect."""

    allow_reuse_address = True
    address_family = socket.AF_INET6

    def server_bind(self):
        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        self.socket.bind(self.server_address)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ThreadingDualStackHTTPServer(ThreadingMixIn, _DualStackAllInterfaces):
    daemon_threads = True


def _make_server(handler: type[Handler], host: str, port: int) -> HTTPServer:
    """Bind so ``http://localhost:…`` (often ::1) and ``http://127.0.0.1:…`` both work."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return ThreadingHTTPServer((host, port), handler)
    try:
        return ThreadingDualStackHTTPServer(("::", port), handler)
    except OSError as e:
        print(
            f"[human_judge_ui] :: bind failed ({e}); using 127.0.0.1 only — "
            f"if API calls fail, open http://127.0.0.1:{port}/ not http://localhost:{port}/",
            file=sys.stderr,
        )
        return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main():
    ap = argparse.ArgumentParser(description="Blind human idea comparison UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help=f"Default LLM for on-demand idea generation (default: {DEFAULT_GENERATION_MODEL}, same as godel_loop)",
    )
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = ap.parse_args()
    if not STATIC_DIR.is_dir():
        print(f"Missing static dir: {STATIC_DIR}", file=sys.stderr)
        sys.exit(1)
    Handler.generation_model = args.model.strip() or DEFAULT_GENERATION_MODEL
    httpd = _make_server(Handler, args.host, args.port)
    if args.host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", ""):
        url = f"http://{args.host}:{args.port}/"
    else:
        url = f"http://127.0.0.1:{args.port}/"
    print(f"Human judge UI: {url}")
    if args.host in ("127.0.0.1", "localhost", "::1"):
        print(f"  (http://localhost:{args.port}/ also works when dual-stack :: bind succeeds)")
    print(f"Scores log: {SCORES_LOG}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
