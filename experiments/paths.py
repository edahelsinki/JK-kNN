"""This file serves as a repository for the project paths."""

from pathlib import Path
import os
from dotenv import dotenv_values

CONFIG = Path(__file__).parent.parent / "config.sh"
cfg = dotenv_values(CONFIG)

ACDB_PATH = cfg.get("ACDB_PATH")

# JKML paths
JKML_PATH = Path(cfg.get("JKCS_PATH")) / "JKML"
