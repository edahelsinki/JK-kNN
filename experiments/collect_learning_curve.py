"""
This script collects a succinct learning curve table from JKML outputs.
Usage:
python3 collect_learning_curve.py [PATH_TO_RESULTS] [SAVE_PATH]

The resulting file will be saved as [SAVE_PATH] / learning_curve.pkl
"""

import pandas as pd
from pathlib import Path
import re
import sys


def collect_results(result_dir: Path, verbose=False):
    """Itreate over result DFs. Assumes a particular file name scheme."""
    if not (result_dir.exists() and result_dir.is_dir()):
        raise FileNotFoundError(f"Directory {result_dir} does not exist!")
    needle = (
        r"^(?:.*?_)?(knn|krr|krr19|mlkr)_(\d+)(?:_([a-zA-Z19-]+))?_(\d+)_trainout\.pkl$"
    )
    dfs = []
    for file in result_dir.glob("*.pkl"):
        match = re.search(needle, str(file.name))
        if match is None:
            continue
        number = match.group(2)
        method = str(match.group(1)).upper()
        if match.group(1) == "krr":
            representation = "FCHL18"
        elif match.group(1) == "krr19":
            representation = "FCHL19"
        else:
            representation = str(match.group(3)).upper()
            sample_size = match.group(4)
        sample_size = match.group(4)
        if verbose:
            print(
                f"{file}: {number} {method} {representation} {sample_size}", flush=True
            )
        if representation == "FCHL":
            representation = "FCHL19"
        df = pd.read_pickle(file)
        if sample_size is None:
            sample_size = len(df.index)
        outdf = {}
        outdf["ml_method"] = method
        outdf["representation"] = representation
        outdf["job"] = number
        outdf["n"] = sample_size
        outdf["error"] = df[("extra", "error")].mean()
        outdf["train_wall"] = df[("extra", "train_wall")].mean()
        outdf["train_cpu"] = df[("extra", "train_cpu")].mean()
        outdf["test_wall"] = df[("extra", "test_wall")].mean()
        outdf["test_cpu"] = df[("extra", "test_cpu")].mean()
        dfs.append(pd.Series(outdf))
    df = pd.DataFrame(dfs)
    df = df.astype({"n": "int32"})
    df.loc[:, "identifier"] = df["ml_method"] + " (" + df["representation"] + ")"
    df.loc[:, "result_dir"] = str(result_dir)
    return df


def collect_clusteromics_results(result_dir: Path, verbose=False):
    if not (result_dir.exists() and result_dir.is_dir()):
        raise FileNotFoundError(f"Directory {result_dir} does not exist!")
    needle = (
        r"^clusteromics_(\w+)_(knn|krr)_(\d+)(?:_([a-zA-Z19-]+))?_(\d+)_trainout\.pkl$"
    )
    dfs = []
    for file in result_dir.glob("*.pkl"):
        match = re.search(needle, str(file.name))
        if match is None:
            continue
        number = match.group(3)
        method = str(match.group(2)).upper()
        if match.group(2) == "krr":
            representation = "FCHL18"
        else:
            representation = str(match.group(4)).upper()
            sample_size = match.group(5)
        sample_size = match.group(5)
        if verbose:
            print(f"{file}: {number} {method} {representation} {sample_size}")
        if representation == "FCHL":
            representation = "FCHL19"
        df = pd.read_pickle(file)
        if sample_size is None:
            sample_size = len(df.index)
        outdf = {}
        outdf["ml_method"] = method
        outdf["representation"] = representation
        outdf["job"] = number
        outdf["n"] = sample_size
        outdf["error"] = df[("extra", "error")].mean()
        outdf["train_wall"] = df[("extra", "train_wall")].mean()
        outdf["train_cpu"] = df[("extra", "train_cpu")].mean()
        outdf["test_wall"] = df[("extra", "test_wall")].mean()
        outdf["test_cpu_total"] = df[("extra", "test_cpu")].mean()
        outdf["test_cpu"] = (df[("extra", "test_cpu")] / df[("extra", "n_test")]).mean()
        outdf["clusteromics_set"] = match.group(1)
        dfs.append(pd.Series(outdf))
    df = pd.DataFrame(dfs)
    df = df.astype({"n": "int32"})
    df.loc[:, "identifier"] = df["ml_method"] + " (" + df["representation"] + ")"
    df.loc[:, "result_dir"] = str(result_dir)
    return df


if __name__ == "__main__":
    res_dir = Path(sys.argv[1])
    if "clusteromics_I-V" in str(res_dir):
        df = collect_clusteromics_results(res_dir, verbose=True)
    else:
        df = collect_results(res_dir, verbose=True)
    fname = res_dir / "learning_curve.pkl"
    with open(fname, "wb") as f:
        df.to_pickle(f)
    print(f"Saved output to {fname}", flush=True)
