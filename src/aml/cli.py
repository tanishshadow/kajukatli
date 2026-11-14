"""Milestone CLI: commands are added only when implemented."""

from pathlib import Path

import typer

from aml.config import Settings
from aml.data.checklist import prepare as prepare_data

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Temporal graph AML backend."""


@app.command()
def prepare(config: Path = typer.Option(Path("configs/default.yaml"), exists=True),
            reuse: bool = typer.Option(False, help="Rebuild the checklist from existing Parquet, without opening raw files.")) -> None:
    """Convert HI-Small to compact Parquet and produce the M1 data checklist."""
    report = prepare_data(Settings.from_yaml(config), reuse=reuse)
    typer.echo(f"Prepared {report['transactions']:,} transactions; {report['positive_transactions']:,} positive.")
    typer.echo(f"Checklist: {Settings.from_yaml(config).output_dir / 'm1' / 'checklist.md'}")


@app.command()
def baseline(config: Path = typer.Option(Path("configs/default.yaml"), exists=True)) -> None:
    """Tune baseline on train and report train/validation metrics."""
    import json
    from aml.eval.run_baseline import run
    typer.echo(json.dumps(run(Settings.from_yaml(config)), indent=2))


@app.command()
def detect(detector: str = typer.Option("fan"), config: Path = typer.Option(Path("configs/default.yaml"), exists=True)) -> None:
    """Run an implemented detector over the full prepared timeline."""
    import json
    from aml.detect.run import run
    typer.echo(json.dumps(run(Settings.from_yaml(config), detector), indent=2))


@app.command()
def narrate(config: Path = typer.Option(Path("configs/default.yaml"), exists=True)) -> None:
    """Produce the M4 evidence-backed backend slice from fan hits."""
    import json
    from aml.narrative.run import run
    typer.echo(json.dumps(run(Settings.from_yaml(config)), indent=2))


@app.command()
def cluster(config: Path = typer.Option(Path("configs/default.yaml"), exists=True)) -> None:
    """Merge evidence, fit train-only models, and export ranked alerts."""
    import json
    from aml.score.run import run
    typer.echo(json.dumps(run(Settings.from_yaml(config)), indent=2))


if __name__ == "__main__":
    app()
