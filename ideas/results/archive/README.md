# Archived result trees (compressed)

Previous evolution loop and dropped workshop artifacts are kept as **tarballs** so the main `results/` tree stays small.

| Archive | Contents |
| -------- | -------- |
| `loop_2026_04_06.tar.gz` | Superseded Gödel loop (S1–S11 era): `results/`, `systems/`, `swe_logs/`, compares, `evolution_log.jsonl`. |
| `workshop_dropped_S20.tar.gz` | S20 candidate: `compare_S15_vs_S20`, blind DeepSeek run, `S20/ideas.json`, `swe_log_S20` (omitted from current workshop tables). |

Restore locally (creates directories; add to `.gitignore` if you do not want them tracked):

```bash
cd ideas/results/archive
tar xzf loop_2026_04_06.tar.gz
tar xzf workshop_dropped_S20.tar.gz
```
