"""Plot training metrics from a PackBalance JSONL log.

Example:
    python scripts/plot_metrics.py \
      --metrics results/tinystories/metrics.jsonl \
      --output results/tinystories/loss_curve.svg \
      --title "TinyStories 49M — packed training (5,000 docs, 1,000 steps)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="PackBalance training")
    parser.add_argument("--ema-span", type=int, default=50)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for line in Path(args.metrics).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    train = [r for r in rows if "loss" in r and "step" in r]
    val = [r for r in rows if "val_loss" in r and "step" in r]
    throughput = [r for r in rows if "tokens_per_sec" in r and "step" in r and r["step"] >= 10]
    if not train:
        raise SystemExit("no training loss records found")

    train_steps = np.asarray([r["step"] for r in train], dtype=np.float64)
    train_loss = np.asarray([r["loss"] for r in train], dtype=np.float64)
    smoothed = _ema(train_loss, args.ema_span)

    fig, (ax_loss, ax_tput) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_loss.plot(
        train_steps,
        train_loss,
        color="#9ecae1",
        linewidth=0.8,
        alpha=0.65,
        label="train (raw)",
    )
    ax_loss.plot(
        train_steps,
        smoothed,
        color="#08519c",
        linewidth=2.0,
        label=f"train (EMA {args.ema_span})",
    )
    if val:
        val_steps = np.asarray([r["step"] for r in val], dtype=np.float64)
        val_loss = np.asarray([r["val_loss"] for r in val], dtype=np.float64)
        ax_loss.plot(
            val_steps,
            val_loss,
            color="#d62728",
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="validation",
        )
        ax_loss.annotate(
            f"final val {val_loss[-1]:.3f}",
            xy=(val_steps[-1], val_loss[-1]),
            xytext=(-70, 18),
            textcoords="offset points",
            color="#d62728",
            fontsize=9,
        )
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title(args.title)
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(loc="upper right")

    if throughput:
        t_steps = np.asarray([r["step"] for r in throughput], dtype=np.float64)
        tps = np.asarray([r["tokens_per_sec"] for r in throughput], dtype=np.float64)
        tps_smooth = _ema(tps, 20)
        ax_tput.plot(t_steps, tps, color="#bcbddc", linewidth=0.8, alpha=0.65)
        ax_tput.plot(t_steps, tps_smooth, color="#54278f", linewidth=1.8, label="tokens/s (EMA 20)")
        ax_tput.axhline(
            float(np.median(tps)),
            color="#54278f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"median {np.median(tps) / 1e3:.1f}k tok/s",
        )
        ax_tput.set_ylabel("tokens/s")
        ax_tput.grid(alpha=0.25)
        ax_tput.legend(loc="lower right", fontsize=8)
    ax_tput.set_xlabel("optimizer step")

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
