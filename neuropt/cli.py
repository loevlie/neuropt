"""CLI entry point: neuropt my_train.py --backend claude"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="neuropt",
    help="LLM-guided ML optimization. Point it at a training script, let it run overnight.",
    add_completion=False,
)


def _load_script(script_path: Path):
    """Import a user's training script and extract train_fn, search_space, model."""
    if not script_path.exists():
        typer.echo(f"Error: {script_path} not found", err=True)
        raise typer.Exit(1)

    spec = importlib.util.spec_from_file_location("user_script", script_path)
    module = importlib.util.module_from_spec(spec)

    # Add script's directory to path so its imports work
    script_dir = str(script_path.parent.resolve())
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec.loader.exec_module(module)

    train_fn = getattr(module, "train_fn", None)
    search_space = getattr(module, "search_space", None)
    model = getattr(module, "model", None)
    ml_context = getattr(module, "ml_context", None)
    minimize = getattr(module, "minimize", None)

    if train_fn is None:
        typer.echo("Error: script must define a 'train_fn' function", err=True)
        raise typer.Exit(1)

    if search_space is None and model is None:
        typer.echo("Error: script must define either 'search_space' (dict) or 'model' (nn.Module)",
                   err=True)
        raise typer.Exit(1)

    return train_fn, search_space, model, ml_context, minimize


@app.command()
def run(
    script: Path = typer.Argument(..., help="Training script (must define train_fn + search_space or model)"),
    backend: str = typer.Option("auto", help="LLM backend: auto, claude, openai, qwen, none"),
    log: str = typer.Option("search.jsonl", help="Log file path (JSONL, supports resume)"),
    batch_size: int = typer.Option(3, "--batch-size", "-b", help="Configs per LLM call"),
    device: Optional[str] = typer.Option(None, help="Device (cuda, mps, cpu). Auto-detects if omitted."),
    timeout: int = typer.Option(600, help="Max seconds per experiment"),
    max_evals: Optional[int] = typer.Option(
        None, "--max-evals", "-n", help="Stop after N experiments (default: run forever)"),
    maximize: bool = typer.Option(
        False, "--maximize", help="Higher scores are better (accuracy, AUROC). Default: minimize (loss)."),
):
    """Run LLM-guided optimization on a training script.

    Your script should define:

      - train_fn(config) -> dict with at least {"score": float}

      - search_space = {"lr": LogUniform(1e-4, 1e-1), ...}
        OR model = nn.Module (auto-introspected)

    Optional: ml_context = "..." to give the LLM domain knowledge.
    Optional: minimize = False if higher scores are better (or pass --maximize).
    """
    from neuropt import ArchSearch

    train_fn, search_space, model, ml_context, script_minimize = _load_script(script)

    # --maximize flag wins; otherwise honor a `minimize` attribute in the script
    if maximize:
        minimize = False
    elif script_minimize is not None:
        minimize = bool(script_minimize)
    else:
        minimize = True

    kwargs = dict(
        backend=backend,
        log_path=log,
        batch_size=batch_size,
        device=device,
        timeout=timeout,
        minimize=minimize,
    )
    if ml_context:
        kwargs["ml_context"] = ml_context

    if model is not None:
        typer.echo(f"Using model introspection from {script.name}")
        search = ArchSearch.from_model(model, train_fn, **kwargs)
    else:
        typer.echo(f"Using search space from {script.name}")
        search = ArchSearch(train_fn=train_fn, search_space=search_space, **kwargs)

    search.run(max_evals=max_evals)


@app.command()
def inspect(
    script: Path = typer.Argument(..., help="Training script with a 'model' variable"),
):
    """Show what neuropt would search over for a given model."""
    from neuropt.introspect import build_search_space, introspect

    _, _, model, _, _ = _load_script(script)
    if model is None:
        typer.echo("No 'model' variable found — nothing to introspect.", err=True)
        typer.echo("This command is for scripts that define model = nn.Module(...).")
        raise typer.Exit(1)

    info = introspect(model)
    space = build_search_space(info)

    typer.echo(f"\nModel: {info['n_params']:,} parameters")
    if info["activation_paths"]:
        typer.echo(f"  Activations: {', '.join(sorted(info['activations_found']))} "
                   f"({len(info['activation_paths'])} layers)")
    if info["has_dropout"]:
        typer.echo(f"  Dropout: {len(info['dropout_paths'])} layers (rate={info['dropout_rate']:.2f})")
    if info["has_batchnorm"]:
        typer.echo(f"  BatchNorm: {len(info['batchnorm_paths'])} layers")
    typer.echo(f"\nSearch space ({len(space)} params):")
    for name, dim in space.items():
        typer.echo(f"  {name}: {dim}")


@app.command()
def results(
    log: Path = typer.Argument("search.jsonl", help="Log file to analyze"),
    top: int = typer.Option(10, help="Show top N results"),
    maximize: bool = typer.Option(
        False, "--maximize", help="Higher scores are better (use if the search ran with maximize)."),
):
    """Show results from a search log."""
    import json

    from neuropt.arch_search import _format_scalars, _get_score

    if not log.exists():
        typer.echo(f"No log file at {log}", err=True)
        raise typer.Exit(1)

    rows = []
    with open(log) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        typer.echo("Empty log file.")
        raise typer.Exit(0)

    worst = float("-inf") if maximize else float("inf")

    def _score(r):
        s = _get_score(r)
        return worst if s is None else s

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=_score, reverse=maximize)

    typer.echo(f"\nTotal experiments: {len(rows)}")
    typer.echo(f"Successful: {len(ok)}")
    if not ok:
        return

    typer.echo(f"\nTop {min(top, len(ok))} results:")
    typer.echo("-" * 70)
    for i, r in enumerate(ok[:top]):
        cfg = r.get("config", {})
        score = _score(r)
        # Show scalars (new format) or legacy keys
        scalars = r.get("scalars", {})
        if scalars:
            extra_parts = _format_scalars(scalars, max_n=4)
        else:
            # Legacy format
            extra_parts = []
            if r.get("val_accuracy"):
                extra_parts.append(f"acc={r['val_accuracy']:.4f}")
            if r.get("n_params"):
                extra_parts.append(f"{r['n_params']:,}p")
        extra = " ".join(extra_parts)
        cfg_keys = list(cfg.keys())[:6]
        cfg_s = ", ".join(f"{k}={cfg[k]}" for k in cfg_keys if k != "device")
        typer.echo(f"  {i+1:>2}. score={score:.4f} {extra}")
        typer.echo(f"      {cfg_s}")

    # Convergence
    typer.echo("\nConvergence:")
    best = worst
    milestones = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    for i, r in enumerate(rows):
        s = _score(r)
        if r.get("status") == "ok" and (s > best if maximize else s < best):
            best = s
        if (i + 1) in milestones:
            typer.echo(f"  After {i+1:>4} evals: {best:.4f}")
    typer.echo(f"  Final ({len(rows):>4} evals): {best:.4f}")


if __name__ == "__main__":
    app()
