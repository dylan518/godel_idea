"""Constants matching EvoScientist (arXiv:2603.08127v1) Sections 3.3 and 4.5.

The open ``ideas/`` harness cannot ship Huawei's closed agent code; this module pins the
paper-reported hyperparameters so experiments can be configured to match the PDF.

Override any value via environment variables where noted.
"""

import os

ARXIV_ID = "2603.08127"
ARXIV_VERSION = "v1"

# --- Section 4.5 Implementation Details ---
# "Scientific idea generation is performed using Gemini-2.5-Pro."
PAPER_IDEA_MODEL = os.environ.get("PAPER_IDEA_MODEL", "gemini-2.5-pro")

# Table 1 / text: automatic idea evaluation used an advanced Gemini judge.
PAPER_IDEA_EVAL_JUDGE = os.environ.get("PAPER_IDEA_EVAL_JUDGE", "gemini-3-flash-preview")

# Literature for stage 1 (paper: Semantic Scholar API).
PAPER_LITERATURE_BACKEND = os.environ.get("PAPER_LITERATURE_BACKEND", "semantic_scholar")

# "we set the ideation retrieval top-k_I to 2"
K_IDEATION = int(os.environ.get("PAPER_K_IDEATION", "2"))

# "maximum of N_I = 21 candidate ideas during idea tree search"
N_I_MAX = int(os.environ.get("PAPER_N_I_MAX", "21"))

# "with 3 parallel workers" (RA) — topic-level parallelism in ``runner.py``.
PARALLEL_WORKERS_RA = int(os.environ.get("PAPER_RA_WORKERS", "3"))

# Memory indexing: "'mxbai-embed-large' embedding model via Ollama"
PAPER_EMBEDDING_MODEL = os.environ.get("PAPER_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_EMBED_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# How many papers to pull for L (paper does not specify a count; 5 matches common setups).
PAPER_LITERATURE_N = int(os.environ.get("PAPER_LITERATURE_N", "5"))
