"""Entry point for `create-benchmark-runner <name>`."""

from pathlib import Path

import click

from benchmark_runner.scaffolder.generator import generate_project, transform_name


DEFAULT_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@click.command()
@click.argument("benchmark_name")
@click.option("--templates-dir", type=click.Path(path_type=Path),
              default=str(DEFAULT_TEMPLATES_DIR),
              help="Templates directory (defaults to the packaged ones).")
def main(benchmark_name: str, templates_dir: Path) -> None:
    """Create a new benchmark runner repo at ./<name>-runner/.

    Example: create-benchmark-runner cyber-bench
    """
    names = transform_name(benchmark_name)
    output_dir = Path.cwd() / f"{names['benchmark_name']}-runner"

    try:
        generate_project(
            benchmark_name=benchmark_name,
            output_dir=output_dir,
            templates_dir=templates_dir,
        )
        click.echo(f"Created {output_dir.name} at {output_dir}")
        click.echo("")
        click.echo("Next steps:")
        click.echo(f"  cd {output_dir.name}")
        click.echo("  make install")
        click.echo("  # edit runner/benchmark.py: implement load_tasks() and generate()")
        click.echo("  # drop your dataset into data/")
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
