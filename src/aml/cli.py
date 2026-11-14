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


if __name__ == "__main__":
    app()
