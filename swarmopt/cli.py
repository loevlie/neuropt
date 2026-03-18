"""CLI entry point: swarmopt my_train.py --backend claude"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="swarmopt",
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

    if train_fn is None:
        typer.echo("Error: script must define a 'train_fn' function", err=True)
        raise typer.Exit(1)

    if search_space is None and model is None:
        typer.echo("Error: script must define either 'search_space' (dict) or 'model' (nn.Module)",
                   err=True)
        raise typer.Exit(1)

    return train_fn, search_space, model, ml_context


@app.command()
def run(
    script: Path = typer.Argument(..., help="Training script (must define train_fn + search_space or model)"),
    backend: str = typer.Option("auto", help="LLM backend: auto, claude, openai, qwen, none"),
    log: str = typer.Option("search.jsonl", help="Log file path (JSONL, supports resume)"),
    batch_size: int = typer.Option(3, "--batch-size", "-b", help="Configs per LLM call"),
    device: Optional[str] = typer.Option(None, help="Device (cuda, mps, cpu). Auto-detects if omitted."),
    timeout: int = typer.Option(600, help="Max seconds per experiment"),
):
    """Run LLM-guided optimization on a training script.

    Your script should define:

      - train_fn(config) -> dict with at least {"score": float}

      - search_space = {"lr": LogUniform(1e-4, 1e-1), ...}
        OR model = nn.Module (auto-introspected)

    Optional: ml_context = "..." to give the LLM domain knowledge.
    """
    from swarmopt import ArchSearch

    train_fn, search_space, model, ml_context = _load_script(script)

    kwargs = dict(
        backend=backend,
        log_path=log,
        batch_size=batch_size,
        device=device,
        timeout=timeout,
    )
    if ml_context:
        kwargs["ml_context"] = ml_context

    if model is not None:
        typer.echo(f"Using model introspection from {script.name}")
        search = ArchSearch.from_model(model, train_fn, **kwargs)
    else:
        typer.echo(f"Using search space from {script.name}")
        search = ArchSearch(train_fn=train_fn, search_space=search_space, **kwargs)

    search.run()


@app.command()
def inspect(
    script: Path = typer.Argument(..., help="Training script with a 'model' variable"),
):
    """Show what swarmopt would search over for a given model."""
    from swarmopt.introspect import introspect, build_search_space

    _, _, model, _ = _load_script(script)
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
):
    """Show results from a search log."""
    import json

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

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("val_loss", float("inf")))

    typer.echo(f"\nTotal experiments: {len(rows)}")
    typer.echo(f"Successful: {len(ok)}")
    if not ok:
        return

    typer.echo(f"\nTop {min(top, len(ok))} results:")
    typer.echo("-" * 70)
    for i, r in enumerate(ok[:top]):
        cfg = r.get("config", {})
        acc = f"acc={r['val_accuracy']:.4f}" if r.get("val_accuracy") else ""
        params = f"{r['n_params']:,}p" if r.get("n_params") else ""
        extra = " ".join(filter(None, [acc, params]))
        cfg_keys = list(cfg.keys())[:6]
        cfg_s = ", ".join(f"{k}={cfg[k]}" for k in cfg_keys if k != "device")
        typer.echo(f"  {i+1:>2}. loss={r['val_loss']:.4f} {extra}")
        typer.echo(f"      {cfg_s}")

    # Convergence
    typer.echo(f"\nConvergence:")
    best = float("inf")
    milestones = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    for i, r in enumerate(rows):
        if r.get("status") == "ok" and r.get("val_loss", float("inf")) < best:
            best = r["val_loss"]
        if (i + 1) in milestones:
            typer.echo(f"  After {i+1:>4} evals: {best:.4f}")
    typer.echo(f"  Final ({len(rows):>4} evals): {best:.4f}")


if __name__ == "__main__":
    app()
