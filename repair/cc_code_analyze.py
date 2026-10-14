#!/usr/bin/env python3
"""compression-confidence Phase 3a analysis: pass@1 on HumanEval+ / MBPP+ per model x precision, and whether the
model's confidence in its code (mean token log-prob) still separates passing from failing solutions.

  python -m repair.cc_code_analyze runs/cc_phase3_a runs/cc_phase3_b
"""
import json
import sys
from pathlib import Path

import numpy as np

from repair.helpers import auroc


def load(cfg_dir, ds):
    res, samp = cfg_dir / f"{ds}_samples_eval_results.json", cfg_dir / f"{ds}_samples.jsonl"
    if not (res.exists() and samp.exists()):
        return None
    ev = json.loads(res.read_text())["eval"]
    conf = {}
    for line in samp.read_text().splitlines():  # keep the FIRST record per task (an overlap wrote duplicates once)
        r = json.loads(line)
        conf.setdefault(r["task_id"], r)
    rows = []
    for tid, lst in ev.items():
        if tid in conf:
            e = lst[0]
            rows.append((e["base_status"] == "pass", e["plus_status"] == "pass", conf[tid]["mean_logprob"],
                         conf[tid]["closed_fence"]))
    return rows


def main():
    dirs = [Path(d) for d in sys.argv[1:]] or [Path("runs/cc_phase3_a"), Path("runs/cc_phase3_b")]
    seen, out = set(), {}
    for root in dirs:
        for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
            for ds in ("humaneval", "mbpp"):
                key = f"{cfg.name}/{ds}"
                if key in seen:
                    continue
                rows = load(cfg, ds)
                if not rows:
                    continue
                seen.add(key)
                base, plus = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
                conf = np.array([r[2] for r in rows])
                keep = np.argsort(-conf, kind="stable")[: int(0.7 * len(rows))]
                out[key] = {"n": len(rows), "base": 100 * base.mean(), "plus": 100 * plus.mean(),
                            "auroc_plus": auroc(conf, plus), "sel70_plus": 100 * plus[keep].mean(),
                            "closed_fence": 100 * np.mean([r[3] for r in rows])}
    Path("runs/cc_phase3_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"{'config':32} {'n':>4} {'base':>6} {'plus':>6} {'AUROC':>6} {'sel@70':>7} {'fence%':>7}")
    for k, r in sorted(out.items(), key=lambda kv: (kv[0].split("/")[1], kv[0])):
        print(f"{k:32} {r['n']:4d} {r['base']:6.1f} {r['plus']:6.1f} {r['auroc_plus']:6.3f} {r['sel70_plus']:7.1f} "
              f"{r['closed_fence']:7.1f}")


if __name__ == "__main__":
    main()
