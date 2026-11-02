"""Turn a Chrome trace into a compact phase/bottleneck report.

This is intentionally independent of TensorBoard: a JSON trace plus a Markdown
summary is enough to answer "where did the step go?" in a terminal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _pct(value: float, total: float) -> float:
    return 100.0 * value / total if total else 0.0


def analyze(trace_path: str | Path) -> dict:
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    phases: dict[str, float] = defaultdict(float)
    steps: list[float] = []
    kernels: dict[str, float] = defaultdict(float)
    cpu_ops: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        cat = event.get("cat", "")
        dur = float(event.get("dur", 0.0))
        if cat == "user_annotation":
            if name.startswith("packbalance/"):
                phases[name] += dur
            elif name.startswith("ProfilerStep"):
                steps.append(dur)
        elif cat == "kernel":
            kernels[name] += dur
        elif cat == "cpu_op":
            cpu_ops[name] += dur
    total_step_ms = sum(steps) / 1e3
    phase_ms = {key: value / 1e3 for key, value in phases.items()}
    accounted = sum(
        phase_ms.get(k, 0.0)
        for k in (
            "packbalance/data",
            "packbalance/block_mask",
            "packbalance/forward",
            "packbalance/backward",
            "packbalance/optimizer",
            "packbalance/comm",
        )
    )
    other_ms = max(0.0, total_step_ms - accounted)
    kernel_ms = {key: value / 1e3 for key, value in kernels.items()}
    total_kernel_ms = sum(kernel_ms.values())
    cpu_ms = {key: value / 1e3 for key, value in cpu_ops.items()}
    total_cpu_ms = sum(cpu_ms.values())
    return {
        "trace": str(trace_path),
        "profiled_steps": len(steps),
        "total_step_ms": total_step_ms,
        "phases_ms": phase_ms,
        "other_ms": other_ms,
        "phase_pct": {k: _pct(v, total_step_ms) for k, v in phase_ms.items()},
        "top_kernels": sorted(kernel_ms.items(), key=lambda kv: kv[1], reverse=True)[:15],
        "kernel_pct": {
            k: _pct(v, total_kernel_ms)
            for k, v in sorted(kernel_ms.items(), key=lambda kv: kv[1], reverse=True)[:15]
        },
        "total_kernel_ms": total_kernel_ms,
        "top_cpu_ops": sorted(cpu_ms.items(), key=lambda kv: kv[1], reverse=True)[:15],
        "total_cpu_ms": total_cpu_ms,
    }


def to_markdown(report: dict) -> str:
    lines = [
        f"# Profile report: `{Path(report['trace']).name}`",
        "",
        f"- Profiled steps: **{report['profiled_steps']}**",
        f"- Wall time: **{report['total_step_ms']:.2f} ms**",
        "",
        "## Step phases",
        "",
        "| phase | ms | % of step |",
        "|---|---:|---:|",
    ]
    for key in (
        "packbalance/data",
        "packbalance/block_mask",
        "packbalance/forward",
        "packbalance/backward",
        "packbalance/optimizer",
        "packbalance/comm",
    ):
        if key in report["phases_ms"]:
            lines.append(
                f"| {key} | {report['phases_ms'][key]:.2f} | {report['phase_pct'][key]:.1f}% |"
            )
    lines.append(
        f"| other | {report['other_ms']:.2f} | {_pct(report['other_ms'], report['total_step_ms']):.1f}% |"
    )
    lines += ["", "## Top GPU kernels", ""]
    if report["top_kernels"]:
        lines += ["| kernel | ms | % of kernel time |", "|---|---:|---:|"]
        for name, ms in report["top_kernels"]:
            lines.append(f"| `{name}` | {ms:.2f} | {report['kernel_pct'][name]:.1f}% |")
        lines.append("")
        lines.append(f"Total kernel time: **{report['total_kernel_ms']:.2f} ms**")
    else:
        lines.append("_CPU-only trace (or no CUDA kernels recorded)._")
    lines += ["", "## Top CPU ops", "", "| op | ms |", "|---|---:|"]
    for name, ms in report["top_cpu_ops"]:
        lines.append(f"| `{name}` | {ms:.2f} |")
    lines.append("")
    lines.append(f"Total CPU op time: **{report['total_cpu_ms']:.2f} ms**")
    lines.append("")
    lines.append(
        "> Phase ranges are `record_function` labels. DataLoader work inside a "
        "worker appears as `packbalance/data` in the main process; CPU preprocessing "
        "is visible in the worker trace only when worker tracing is enabled."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default="artifacts/profile_report.md")
    args = parser.parse_args()
    report = analyze(args.trace)
    markdown = to_markdown(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown)
    print(f"\nwrote {output} and {json_path}")


if __name__ == "__main__":
    main()
