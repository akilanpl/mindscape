from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)
