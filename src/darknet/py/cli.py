"""Console script for darknet.py."""

import sys
import click


@click.command()
def py(args=None):
    """Console script for darknet.py."""
    # fmt: off
    click.echo("Replace this message by putting your code into "
               "py.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    # fmt: on
    return 0


if __name__ == "__main__":
    sys.exit(py)  # pragma: no cover