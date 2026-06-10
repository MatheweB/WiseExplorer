"""Thin wrapper — the demo is the `transfer` CLI verb.

    wise-explorer transfer            # discover on 4-pile Nim, play 8-pile zero-shot
    wise-explorer transfer --full     # + the honest controls
    wise-explorer transfer --piles 10 # a bigger target

This file just forwards so `python scripts/transfer_demo.py` keeps working.
"""
import sys

from wise_explorer.cli import run_transfer

if __name__ == "__main__":
    run_transfer(sys.argv[1:])
