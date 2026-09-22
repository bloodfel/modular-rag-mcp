"""Render benchmark report charts from a rerank_benchmark JSON file.

Reads the per-combination aggregates written by scripts/benchmark_rerank.py
and overwrites reports/images/{ndcg,jev_gain,latency}.png.

Usage:
    .venv/bin/python scripts/make_charts.py reports/rerank_benchmark_2026-09-22.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed palette so each reranker keeps the same color across every chart.
COLORS = {"none": "#9CA3AF", "bge": "#3B82F6", "jev": "#F97316", "llm": "#10B981"}
LABELS = {"none": "none（不重排）", "bge": "bge（本地）", "jev": "jev（API）", "llm": "GLM-4-Flash（API）"}
DPI = 200

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_rows(path: Path) -> list[dict]:
    """Load combination aggregates from a benchmark JSON report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    combos = data.get("combinations") or data.get("results") or []
    if not combos and isinstance(data, list):
        combos = data
    return combos


def row_key(row: dict) -> str:
    return f"{row['dataset']}/{row['retriever']}"


def grouped_bars(rows: list[dict], metric: str, title: str, ylabel: str, out: Path,
                 scale: float = 1.0, log_y: bool = False, fmt: str = "%.2f") -> None:
    """Draw a grouped bar chart: x = dataset/retriever, series = reranker."""
    labels = sorted({row_key(r) for r in rows})
    rerankers = [r for r in ["none", "bge", "jev", "llm"] if any(x["reranker"] == r for x in rows)]
    lookup = {(row_key(r), r["reranker"]): r[metric] for r in rows}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / len(rerankers)
    for i, rer in enumerate(rerankers):
        xs = [j + (i - (len(rerankers) - 1) / 2) * width for j in range(len(labels))]
        vals = [lookup.get((lab, rer)) for lab in labels]
        bars = ax.bar(xs, [v * scale for v in vals], width * 0.92,
                      label=LABELS[rer], color=COLORS[rer], zorder=3)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=7.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    if log_y:
        ax.set_yscale("log")
        ax.set_axisbelow(True)
    else:
        ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def gain_bars(rows: list[dict], out: Path) -> None:
    """Jev gain over no-rerank per dataset/retriever (nDCG@10 points x100)."""
    labels = sorted({row_key(r) for r in rows})
    lookup = {(row_key(r), r["reranker"]): r["ndcg_at_10"] for r in rows}
    gains = [lookup[(lab, "jev")] - lookup[(lab, "none")] for lab in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16A34A" if g >= 0 else "#DC2626" for g in gains]
    bars = ax.bar(labels, [g * 100 for g in gains], 0.55, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%+.2f", padding=2, fontsize=9)

    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_ylabel("Δ nDCG@10（×100）", fontsize=10)
    ax.set_title("Jev 相对不重排的质量增益", fontsize=12, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(top=max(gains) * 100 + 2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/rerank_benchmark_2026-09-22.json")
    outdir = Path("reports/images")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(src)
    if not rows:
        print(f"no combinations found in {src}")
        return 1

    grouped_bars(
        rows, "ndcg_at_10",
        "重排质量对比：nDCG@10（×100，越高越好）",
        "nDCG@10 ×100", outdir / "ndcg.png", scale=100.0,
    )
    gain_bars(rows, outdir / "jev_gain.png")
    grouped_bars(
        rows, "rerank_ms_p50",
        "单次重排延迟 p50（秒，越低越好，对数轴）",
        "latency p50 (s)", outdir / "latency.png",
        scale=0.001, log_y=True, fmt="%.2fs",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
