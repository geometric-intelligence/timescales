#!/usr/bin/env python3
"""Convert a jupytext percent-format .py into a .ipynb, with no extra dependencies.

The repo stores notebooks as percent-format .py (diffable, reviewable). Jupytext is not
installed in every environment, so this emits the .ipynb directly.

Usage:
    python _to_ipynb.py 1_linear_sweep_analysis.py
"""

import json
import os
import sys


def parse_cells(text: str):
    """Split percent-format source into (cell_type, source) pairs."""
    cells, kind, buf = [], "code", []

    def flush():
        if buf and "".join(buf).strip():
            cells.append((kind, "".join(buf).rstrip("\n")))
        buf.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        if kind == "markdown":
            # Markdown cells are stored as comments; strip the leading "# ".
            if stripped.startswith("# "):
                buf.append(stripped[2:] + "\n")
            elif stripped == "#":
                buf.append("\n")
            else:
                buf.append(line)
        else:
            buf.append(line)
    flush()
    return cells


def to_notebook(cells):
    out = []
    for kind, src in cells:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (timescales)",
                "language": "python",
                "name": "timescales",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src_path = sys.argv[1]
    with open(src_path) as f:
        nb = to_notebook(parse_cells(f.read()))
    dst = os.path.splitext(src_path)[0] + ".ipynb"
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {dst} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
