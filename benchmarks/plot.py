"""Figures and tables from the cached harness runs.

    python -m benchmarks.plot            # writes benchmarks/figures/*.png and benchmarks/results.md

Reported score: the test loss of the config with the best validation loss so far (the incumbent).
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

from benchmarks.run import RUNS_DIR
from benchmarks.tasks import TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
MILESTONES = [5, 10, 15, 30]

# base method -> (label, color, marker); "+default" variants are drawn dashed in the same color
STYLE = {
    "random": ("Random search", "#8a8980", "x"),
    "tpe": ("Optuna TPE", "#eda100", "v"),
    "neuropt-haiku-4.5": ("neuropt, Claude Haiku 4.5", "#2a78d6", "o"),
    "neuropt-gpt-4o-mini": ("neuropt, GPT-4o mini", "#1baf7a", "s"),
    "jev": ("neuropt, Jev", "#eb6834", "^"),
    # neuropt-v2-*: this branch's ArchSearch (structured outputs, partial batches kept, 20-point curves)
    "neuropt-v2-haiku-4.5": ("neuropt v2, Claude Haiku 4.5", "#0b3f8c", "o"),
    "neuropt-v2-haiku-5.5": ("neuropt v2, Claude Haiku 5.5", "#8a3fd1", "D"),
    "neuropt-v2-gpt-4o-mini": ("neuropt v2, GPT-4o mini", "#0e7a52", "s"),
    "neuropt-v2-qwen3.5-9b": ("neuropt v2, Qwen3.5 9B (local)", "#c2367f", "P"),
    "hybrid-v2-haiku-5.5": ("Haiku 5.5 then TPE", "#4d4d4d", "*"),
}
# (variant, baseline) pairs judged by the decision rule in docs/evals.md
COMPARISONS = [
    ("neuropt-haiku-4.5", "tpe"), ("neuropt-haiku-4.5", "random"),
    ("neuropt-haiku-4.5+default", "tpe+default"), ("neuropt-haiku-4.5+default", "random+default"),
    ("neuropt-gpt-4o-mini", "tpe"), ("neuropt-gpt-4o-mini+default", "tpe+default"),
    ("jev", "tpe"), ("jev", "random"), ("jev+default", "tpe+default"),
    ("jev+default", "random+default"), ("jev", "neuropt-haiku-4.5"),
    ("tpe+default", "tpe"), ("neuropt-haiku-4.5+default", "neuropt-haiku-4.5"),
    # Phase 2: same model, new proposal handling (v2) against Phase 1 (0.6.1)
    ("neuropt-v2-haiku-4.5", "neuropt-haiku-4.5"),
    ("neuropt-v2-haiku-4.5+default", "neuropt-haiku-4.5+default"),
    ("neuropt-v2-gpt-4o-mini", "neuropt-gpt-4o-mini"),
    ("neuropt-v2-gpt-4o-mini+default", "neuropt-gpt-4o-mini+default"),
    # Phase 2/3: v2 arms against the baselines and each other
    *[(f"{m}{s}", f"tpe{s}") for m in ("neuropt-v2-haiku-4.5", "neuropt-v2-haiku-5.5",
                                       "neuropt-v2-gpt-4o-mini", "neuropt-v2-qwen3.5-9b",
                                       "hybrid-v2-haiku-5.5") for s in ("", "+default")],
    ("neuropt-v2-haiku-5.5", "neuropt-v2-haiku-4.5"),
    ("neuropt-v2-haiku-5.5+default", "neuropt-v2-haiku-4.5+default"),
    ("hybrid-v2-haiku-5.5", "neuropt-v2-haiku-5.5"),
    ("hybrid-v2-haiku-5.5+default", "neuropt-v2-haiku-5.5+default"),
]


def load_runs(runs_dir=RUNS_DIR):
    """{task: {method: [run, ...]}} for complete runs; a run is {"rows", "summary", curves...}."""
    runs = defaultdict(lambda: defaultdict(list))
    for root, _, files in os.walk(runs_dir):
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(root, name)) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if not rows or rows[-1].get("type") != "summary":
                continue
            summary, evals = rows[-1], rows[:-1]
            best, val_curve, test_curve = None, [], []
            for r in evals:
                if best is None or r["val_score"] < best["val_score"]:
                    best = r
                val_curve.append(best["val_score"])
                test_curve.append(best["test_score"])
            keys = [json.dumps(r["config"], sort_keys=True) for r in evals]
            runs[summary["task"]][summary["method"]].append({
                "seed": summary["seed"], "summary": summary, "rows": evals,
                "val": val_curve, "test": test_curve,
                "duplicate_rate": 1 - len(set(keys)) / len(keys),
                "failed_evals": sum(r["status"] != "ok" for r in evals),
            })
    for task in runs.values():
        for method_runs in task.values():
            method_runs.sort(key=lambda r: r["seed"])
    return runs


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _fmt(values, digits=3):
    mean, std = _mean_std(values)
    return "n/a" if mean is None else f"{mean:.{digits}f} ± {std:.{digits}f}"


def compare(runs_a, runs_b, at):
    """Decision rule: A helps over B if its mean is lower and it wins on >= 4 of 5 paired seeds."""
    a = {r["seed"]: r["test"][at - 1] for r in runs_a if len(r["test"]) >= at}
    b = {r["seed"]: r["test"][at - 1] for r in runs_b if len(r["test"]) >= at}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    diff = statistics.mean(a[s] - b[s] for s in seeds)
    wins = sum(a[s] < b[s] for s in seeds)
    losses = sum(a[s] > b[s] for s in seeds)
    if len(seeds) < 5:
        verdict = f"incomplete ({len(seeds)} of 5 seeds)"
    elif diff < 0 and wins >= 4:
        verdict = "helps"
    elif diff > 0 and losses >= 4:
        verdict = "worse"
    else:
        verdict = "no measurable difference"
    return {"n": len(seeds), "diff": diff, "wins": wins, "losses": losses, "verdict": verdict}


def _order(methods):
    known = [m + s for m in STYLE for s in ("", "+default")]
    return [m for m in known if m in methods] + sorted(m for m in methods if m not in known)


def write_tables(runs, path):
    lines = ["# Harness results", "",
             "Generated by `python -m benchmarks.plot`. Scores are the **test** loss of the config with "
             "the best validation loss so far, mean ± std over seeds. Lower is better.", ""]
    for task_name in [t for t in TASKS if t in runs]:
        methods = runs[task_name]
        n_evals = max(len(r["test"]) for rs in methods.values() for r in rs)
        marks = [m for m in MILESTONES if m <= n_evals] or [n_evals]
        lines += [f"## {TASKS[task_name].title}", ""]
        hosts = sorted({r["summary"]["host"] for rs in methods.values() for r in rs})
        lines += [f"Hardware: {', '.join(hosts)}. Evals per run: {n_evals}.", ""]
        lines.append("| Method | Seeds | " + " | ".join(f"Test loss @{m}" for m in marks)
                     + f" | Val loss @{marks[-1]} | Test acc @{marks[-1]} |")
        lines.append("|---|:---:|" + ":---:|" * (len(marks) + 2))
        for method in _order(methods):
            rs = methods[method]
            cells = [_fmt([r["test"][m - 1] for r in rs]) for m in marks]
            val = _fmt([r["val"][marks[-1] - 1] for r in rs])
            accs = []
            for r in rs:
                best = min(r["rows"][:marks[-1]], key=lambda row: row["val_score"])
                accs.append(best["test_accuracy"])
            lines.append(f"| `{method}` | {len(rs)} | " + " | ".join(cells)
                         + f" | {val} | {_fmt(accs)} |")
        lines += ["", "| Method | Wall time (s) | Training time (s) | LLM cost ($) | LLM calls | "
                      "Latency per call (s) | Tokens in / out | Parse failures | LLM errors | "
                      "Duplicate proposals | Fallback evals | Failed evals | Repeated configs |",
                  "|---|" + ":---:|" * 12]
        for method in _order(methods):
            rs = methods[method]
            llm = [r["summary"]["llm"] or {} for r in rs]

            def total(key, llm=llm):
                return sum(x.get(key) or 0 for x in llm)

            has_llm = any(llm)
            lines.append(
                f"| `{method}` | {_fmt([r['summary']['wall_time'] for r in rs], 0)} | "
                f"{_fmt([r['summary']['train_time'] for r in rs], 0)} | "
                + (f"{_fmt([x.get('cost_usd') for x in llm], 4)} | {total('calls')} | "
                   f"{_fmt([x.get('latency_mean') for x in llm], 2)} | "
                   f"{total('tokens_in'):,} / {total('tokens_out'):,} | "
                   + (f"{total('parse_failures')} of {total('llm_calls')} | {total('llm_errors')} | "
                      f"{total('duplicate_configs')} of {total('proposed_configs')} | "
                      if total("llm_calls") else "n/a | n/a | n/a | ")  # decision models don't write JSON
                   +
                   f"{total('fallback_evals')} | " if has_llm else "none | " + "n/a | " * 7)
                + f"{sum(r['failed_evals'] for r in rs)} | "
                f"{100 * statistics.mean(r['duplicate_rate'] for r in rs):.1f}% |")
        lines += ["", f"Paired comparison at {marks[-1]} evals (decision rule in `docs/evals.md`):", "",
                  "| Variant | Baseline | Mean test-loss difference | Seeds better / worse | Verdict |",
                  "|---|---|:---:|:---:|---|"]
        for a, b in COMPARISONS:
            if a in methods and b in methods:
                c = compare(methods[a], methods[b], marks[-1])
                if c:
                    lines.append(f"| `{a}` | `{b}` | {c['diff']:+.4f} | {c['wins']} / {c['losses']} "
                                 f"of {c['n']} | {c['verdict']} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot(runs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    ink, muted, grid = "#1f1f1d", "#6b6a63", "#e6e4dc"
    os.makedirs(out_dir, exist_ok=True)
    tasks = [t for t in TASKS if t in runs]

    def draw(ax, task_name, key, seeded):
        tops, n = [], 0
        for method in _order(runs[task_name]):
            base, _, suffix = method.partition("+")
            if base not in STYLE or bool(suffix) != seeded:
                continue
            label, color, marker = STYLE[base]
            curves = np.array([r[key] for r in runs[task_name][method]])
            mean = curves.mean(0)
            std = curves.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            x = np.arange(1, len(mean) + 1)
            n = len(mean)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, lw=0)
            ax.plot(x, mean, color=color, lw=2, marker=marker, ms=4.5, markevery=5, label=label)
            tops.append(mean[min(4, n - 1)])  # value at eval 5: keeps the early spike off-scale
        ax.set_xlim(1, n)
        ax.set_xticks([1] + [m for m in MILESTONES if m <= n])
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_color(grid)
        return tops

    rows = (("Search from scratch", False), ("Default config evaluated first", True))
    for key, name, ylabel in (("test", "test", "Test loss of the best-validation config"),
                              ("val", "val", "Best validation loss so far")):
        fig, axes = plt.subplots(2, len(tasks), figsize=(4.9 * len(tasks), 7.6), squeeze=False,
                                 sharey="col")
        for col, task_name in enumerate(tasks):
            tops = []
            for row, (row_title, seeded) in enumerate(rows):
                ax = axes[row][col]
                tops += draw(ax, task_name, key, seeded)
                ax.set_title(f"{TASKS[task_name].title}\n{row_title}", loc="left", color=ink,
                             fontsize=10, fontweight="bold")
            low = min(statistics.mean(r[key][-1] for r in rs) for rs in runs[task_name].values())
            span = max(tops) - low
            axes[0][col].set_ylim(low - 0.12 * span, max(tops) + 0.10 * span)
            n_seeds = min(len(rs) for rs in runs[task_name].values())
            axes[1][col].set_xlabel(f"Evaluations used (mean of {n_seeds} seeds, band = ± 1 std)",
                                    color=muted)
        for row in range(2):
            axes[row][0].set_ylabel(ylabel + "\n(lower is better)", color=muted)
        entries = {}
        for ax in axes.flat:  # a method may be missing from some tasks
            for handle, label in zip(*ax.get_legend_handles_labels()):
                entries.setdefault(label, handle)
        fig.legend(entries.values(), entries.keys(), loc="lower center", ncol=5, frameon=False,
                   fontsize=9.5)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(os.path.join(out_dir, f"harness_{name}.png"), dpi=160)
        plt.close(fig)


def plot_extra(runs, out_dir):
    """Three more views: per-seed differences to TPE, score against LLM cost, and fallback rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, muted, grid = "#1f1f1d", "#6b6a63", "#e6e4dc"
    tasks = [t for t in TASKS if t in runs]
    llm = [m for m in STYLE if m not in ("random", "tpe")]

    def style(ax, title):
        ax.set_title(title, loc="left", color=ink, fontsize=10, fontweight="bold")
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_color(grid)

    def final(task_name, method):
        return {r["seed"]: r["test"][-1] for r in runs[task_name].get(method, [])}

    # 1. Per-seed difference to Optuna TPE with the same seeding
    arms = [m + suffix for m in llm for suffix in ("", "+default")]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.9 * len(tasks), 4.4), squeeze=False, sharey=True)
    for ax, task_name in zip(axes[0], tasks):
        for y, arm in enumerate(reversed(arms)):
            base, _, suffix = arm.partition("+")
            a, b = final(task_name, arm), final(task_name, "tpe" + ("+default" if suffix else ""))
            diffs = [a[s] - b[s] for s in sorted(set(a) & set(b))]
            if not diffs:
                continue
            color = STYLE[base][1]
            jitter = np.linspace(-0.16, 0.16, len(diffs))
            ax.scatter(diffs, y + jitter, s=42, color=color if not suffix else "white",
                       edgecolor=color, linewidth=1.6, zorder=3)
            ax.plot([np.mean(diffs)] * 2, [y - 0.32, y + 0.32], color=ink, lw=2, zorder=4)
        ax.axvline(0, color=muted, lw=1)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([STYLE[m.partition("+")[0]][0].replace("neuropt, ", "")
                            + (", default first" if "+" in m else "") for m in reversed(arms)])
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_xlabel("Test loss minus Optuna TPE's, same seed\n(left of 0 = neuropt better)", color=muted)
        style(ax, TASKS[task_name].title)
    fig.suptitle("Each dot is one seed after 30 evals; the bar is the mean. Hollow dots: default config "
                 "evaluated first, compared with default-seeded TPE.", x=0.01, ha="left", fontsize=9.5,
                 color=muted)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "harness_paired.png"), dpi=160)
    plt.close(fig)

    # 2. Final test loss against LLM cost per run
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.9 * len(tasks), 4.4), squeeze=False)
    for ax, task_name in zip(axes[0], tasks):
        for suffix, ls in (("", "-"), ("+default", "--")):
            ref = list(final(task_name, "tpe" + suffix).values())
            if ref:
                ax.axhline(np.mean(ref), color=STYLE["tpe"][1], lw=1.6, ls=ls,
                           label="Optuna TPE" + (", default first" if suffix else "") + " (no LLM cost)")
        for base in llm:
            for suffix in ("", "+default"):
                rs = runs[task_name].get(base + suffix, [])
                if not rs:
                    continue
                cost = np.mean([(r["summary"]["llm"] or {}).get("cost_usd") or 0 for r in rs])
                vals = [r["test"][-1] for r in rs]
                color = STYLE[base][1]
                ax.errorbar(cost, np.mean(vals), yerr=np.std(vals, ddof=1), color=color, lw=1.2,
                            marker=STYLE[base][2], ms=8, mfc="white" if suffix else color, mec=color,
                            mew=1.6, capsize=3,
                            label=STYLE[base][0] + (", default first" if suffix else ""))
        ax.set_xscale("log")
        ax.set_xlim(2e-3, 2)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_xlabel("LLM cost per 30-eval run (USD, log scale)", color=muted)
        style(ax, TASKS[task_name].title)
    axes[0][0].set_ylabel("Test loss after 30 evals\n(mean ± std over seeds, lower is better)", color=muted)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(out_dir, "harness_cost.png"), dpi=160)
    plt.close(fig)

    # 3. Share of evals that fell back to random configs because the model's answer was unusable
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.8 / len(llm)
    for i, base in enumerate(llm):
        shares = []
        for task_name in tasks:
            rs = runs[task_name].get(base, []) + runs[task_name].get(base + "+default", [])
            evals = sum(len(r["rows"]) for r in rs)
            fallback = sum((r["summary"]["llm"] or {}).get("fallback_evals", 0) for r in rs)
            shares.append(100 * fallback / evals if evals else 0)
        x = np.arange(len(tasks)) + (i - (len(llm) - 1) / 2) * width
        bars = ax.bar(x, shares, width * 0.9, color=STYLE[base][1], label=STYLE[base][0])
        for rect, share in zip(bars, shares):
            ax.text(rect.get_x() + rect.get_width() / 2, share + 0.4, f"{share:.0f}%", ha="center",
                    va="bottom", fontsize=9, color=ink)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASKS[t].title.split(" (")[0] for t in tasks])
    ax.set_ylabel("Evals run as random fallback (%)", color=muted)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    style(ax, "How often the model's answer was unusable (10 runs x 30 evals per bar)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "harness_fallback.png"), dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    parser.add_argument("--tables", default=os.path.join(HERE, "results.md"))
    args = parser.parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"no complete runs in {args.runs_dir}")
        return 1
    write_tables(runs, args.tables)
    plot(runs, args.out_dir)
    plot_extra(runs, args.out_dir)
    print(f"wrote {args.tables} and figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
