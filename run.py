#!/usr/bin/env python3
"""
P-02 harness driver (organizer-style CLI).

The official kit places ``harness.py``, ``generator.py``, ``schema.py``, … next to
``run.py``. For local runs from this repo only, set:

  export P02_BENCH=/absolute/path/to/p02-harness-directory

The contexter repo does **not** ship ``harness.py``; either merge this tree with the official
bench (so ``harness.py`` sits next to ``run.py``), or point at the harness directory:

  export P02_BENCH=/absolute/path/to/p02-harness-directory

Or pass ``--bench`` once (same path) when running from the contexter repo only.

Then:

  python run.py --adapter adapters.myteam:Engine --mode fast \\
    --seeds 9999 31415 27182 16180 11235 \\
    --n-services 20 --days 14 --out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _bench_dir(cli_bench: Path | None) -> Path:
    here = _repo_root()
    if cli_bench is not None:
        p = cli_bench.expanduser().resolve()
        if (p / "harness.py").is_file():
            return p
        raise SystemExit(
            f"--bench {p} does not contain harness.py (wrong directory or typo)."
        )
    if (here / "harness.py").is_file():
        return here
    env = os.environ.get("P02_BENCH") or os.environ.get("P02_HARNESS_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "harness.py").is_file():
            return p
    raise SystemExit(
        "Cannot find harness.py next to run.py and P02_BENCH / P02_HARNESS_ROOT is unset or invalid.\n"
        "The contexter directory only contains run.py + your adapter — the P-02 harness lives in a "
        "separate kit (harness.py, generator.py, schema.py, …).\n\n"
        "Fix one of:\n"
        "  1) export P02_BENCH=/path/to/that/harness-directory\n"
        "  2) python run.py --bench /path/to/that/harness-directory ... (same path)\n"
        "  3) Copy/merge the harness files next to run.py, then run from that folder.\n"
    )


def _adapter_factory_from_spec(spec: str) -> Callable[[], Any]:
    """Load ``adapters.<name>:Class`` from this repo without shadowing the harness ``adapter`` module."""
    if ":" not in spec:
        raise SystemExit(f"Invalid --adapter (expected module:Class): {spec!r}")
    mod_path, cls_name = spec.rsplit(":", 1)
    parts = mod_path.split(".")
    repo = _repo_root()
    if len(parts) != 2 or parts[0] != "adapters":
        raise SystemExit(
            f"Unsupported --adapter {spec!r}. Use adapters.<module>:<Class> "
            f"(file: {repo / 'adapters'}/<module>.py)."
        )
    stem = parts[1]
    path = repo / "adapters" / f"{stem}.py"
    if not path.is_file():
        raise SystemExit(f"Adapter file not found: {path}")
    alias = f"_p02_loaded_adapter_{stem}"
    s = importlib.util.spec_from_file_location(alias, path)
    if s is None or s.loader is None:
        raise SystemExit(f"Cannot load adapter from {path}")
    m = importlib.util.module_from_spec(s)
    sys.modules[alias] = m
    s.loader.exec_module(m)
    cls = getattr(m, cls_name)
    return lambda: cls()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-02 harness run (organizer CLI shape)")
    ap.add_argument("--adapter", required=True, help="e.g. adapters.myteam:Engine")
    ap.add_argument(
        "--bench",
        type=Path,
        default=None,
        help="Directory with harness.py (use from contexter repo; else set P02_BENCH).",
    )
    ap.add_argument("--mode", choices=["fast", "deep"], default="fast")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-services", type=int, required=True)
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Write full JSON report here")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup queries per seed (default 2)")
    args = ap.parse_args(argv)

    bench = _bench_dir(args.bench)
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))

    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(1, str(repo))

    from generator import GenConfig
    from harness import run

    factory = _adapter_factory_from_spec(args.adapter)
    base_seed = args.seeds[0]
    cfg = GenConfig(seed=base_seed, n_services=args.n_services, days=args.days)

    t0 = time.monotonic()
    report = run(factory, cfg, mode=args.mode, seeds=list(args.seeds), warmup=args.warmup)
    wall_ms = (time.monotonic() - t0) * 1000.0
    report["total_wall_time_ms"] = round(wall_ms, 2)
    report["cli"] = {
        "adapter": args.adapter,
        "mode": args.mode,
        "seeds": list(args.seeds),
        "n_services": args.n_services,
        "days": args.days,
        "warmup": args.warmup,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    agg = report.get("aggregated") or {}
    sc = report.get("score") or {}
    print(f"Wrote {args.out.resolve()}")
    print(
        f"  recall@5={agg.get('recall@5')}  precision@5_mean={agg.get('precision@5_mean')}  "
        f"remediation_acc={agg.get('remediation_acc')}  "
        f"weighted={sc.get('weighted_score')}  wall_ms={wall_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
