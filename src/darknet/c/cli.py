"""Console script for darknet.c."""

import sys
import click


@click.command()
def c(args=None):
    """Console script for darknet.c."""
    # fmt: off
    click.echo("Replace this message by putting your code into "
               "c.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    # fmt: on
    return 0


if __name__ == "__main__":
    sys.exit(c)  # pragma: no cover