"""
This script constructs a k-sensitivity dataset based on pretrained k-NN models.
Usage:
python3 k_sensitivity.py [PATH_TO_RESULTS] [SAVE_PATH]

The resulting file will be saved as [SAVE_PATH] / k_sensitivity_i.pkl
"""

import pandas as pd
from pathlib import Path
import re
from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase import Atoms
import pickle
import numpy as np
from typing import Iterable, Dict, Union
from sklearn.metrics import pairwise_distances
from constants import *
import paths
import sys
import time

sys.path.append(paths.JKML_PATH)
from src.QKNN import (
    VPTreeKNN18,
    load_fchl18_vp_knn,
    VPTreeKNN19,
    load_fchl19_vp_knn,
    correct_fchl18_kernel_size,
)


def _generate_fchl19(
    strs: Iterable[Atoms], max_atoms=None, elements=None, rcut=8.0, acut=8.0, **kwargs
) -> np.ndarray:
    from qmllib.representations import generate_fchl19 as generate_representation

    if elements is None:
        elements = [1, 6, 7, 8, 16]
    if max_atoms is None:
        max_atoms = max([len(s.get_atomic_numbers()) for s in strs])
    n = len(strs)
    representation = generate_representation(
        strs[0].get_atomic_numbers(),
        strs[0].get_positions(),
        elements=elements,
        rcut=rcut,
        acut=acut,
        pad=max_atoms,
    )
    X = np.zeros((n, representation.shape[1]))
    X[0, :] = np.sum(representation, axis=0)
    for i in range(1, n):
        X[i, :] = generate_representation(
            strs[i].get_atomic_numbers(),
            strs[i].get_positions(),
            elements=elements,
            rcut=rcut,
            acut=acut,
            pad=max_atoms,
        ).sum(axis=0)
    if np.isnan(X).any():
        raise ValueError("NaNs in FCHL representation!")
    return X


def _generate_mbdf(
    strs: Iterable[Atoms], max_atoms=None, cutoff: float = 8.0, **kwargs
) -> np.ndarray:
    from MBDF import generate_mbdf as generate_representation

    if max_atoms is None:
        max_atoms = max([len(s.get_atomic_numbers()) for s in strs])
    n = len(strs)
    ragged_atomic_numbers = np.empty(n, dtype=object)
    ragged_atomic_numbers[:] = [i.get_atomic_numbers() for i in strs]
    ragged_positions = np.empty(n, dtype=object)
    ragged_positions[:] = [i.get_positions() for i in strs]
    X = generate_representation(
        ragged_atomic_numbers,
        ragged_positions,
        cutoff_r=cutoff,
        normalized=False,
        local=False,
        pad=max_atoms,
    )
    return X


def _generate_fchl18(strs: Iterable[Atoms], max_atoms=None, cutoff=8.0):
    from qmllib.representations import generate_fchl18 as generate_representation

    if max_atoms is None:
        max_atoms = max([len(s.get_atomic_numbers()) for s in strs])

    representations = []
    for struct in strs:
        representations.append(
            generate_representation(
                struct.get_atomic_numbers(),
                struct.get_positions(),
                max_size=max_atoms,
                neighbors=max_atoms,
                cut_distance=cutoff,
            )
        )
    return np.array(representations)


def calculate_representation(Qrepresentation, strs, **repr_kwargs):
    if Qrepresentation in ["fchl", "fchl-nometric"]:
        return _generate_fchl19(strs, **repr_kwargs)
    elif Qrepresentation in ["mbdf", "mbdf-nometric"]:
        return _generate_mbdf(strs, **repr_kwargs)
    elif "fchl-kernel" in Qrepresentation:
        return _generate_fchl18(strs, **repr_kwargs)
    else:
        raise NotImplementedError(
            f"Representation 'f{Qrepresentation}' not supported with the k-NN model!"
        )


def parse_fname(model_path: Path):
    needle = r"^(?:.*?_)?(knn|krr|mlkr)_(\d+)(?:_([a-zA-Z-]+))?_(\d+)\.pkl$"
    re_match = re.search(needle, str(model_path.name))
    if re_match is None:
        return None
    job_id = re_match.group(2)
    method = str(re_match.group(1)).upper()
    if method == "KRR":
        representation = "FCHL18"
    else:
        representation = str(re_match.group(3))
    if "-nometric" in representation:
        representation = representation.split("-nometric")[0]
        no_metric = True
    else:
        no_metric = False
    sample_size = re_match.group(4)
    return job_id, method, representation, no_metric, sample_size


def load_test_data(test_data_path: Path):
    with open(test_data_path, "rb") as f:
        f.seek(0)
        df_test = pd.read_pickle(f)
    return df_test


def load_model(path_stem: Path):
    model_path = path_stem.with_suffix(".pkl")
    job_id, method, representation, no_metric, sample_size = parse_fname(model_path)
    with open(model_path, "rb") as f:
        if no_metric:
            X_train, Y_train, X_atoms, knn_params, train_metadata = pickle.load(f)
        elif "fchl-kernel" in representation:
            (
                X_train,
                Y_train,
                X_atoms,
                knn_params,
                vp_params,
                train_metadata,
            ) = pickle.load(f)
        else:
            (
                X_train,
                Y_train,
                X_atoms,
                A,
                mlkr,
                knn_params,
                train_metadata,
            ) = pickle.load(f)

    # need to recreate the model due to not being able to pickle the custom metric
    if not no_metric and ("fchl19-kernel" not in representation):
        knn_params["metric"] = mlkr.get_metric()
    if representation == "fchl19-kernel":

        knn = load_fchl19_vp_knn(X_train, Y_train, vp_params, **knn_params)
    else:
        knn = KNeighborsRegressor(**knn_params)
        knn.fit(X_train, Y_train)
    out = {"knn": knn, "X_train": X_train, "Y_train": Y_train}
    if ((representation == "fchl") or (representation == "mbdf")) and not no_metric:
        out["mlkr"] = mlkr
    out["Qrepresentation"] = representation
    out["no_metric"] = bool(no_metric)
    out["X_train"] = X_train
    out["Y_train"] = Y_train
    out["method"] = method
    out["sample_size"] = int(sample_size)
    out["job_id"] = int(job_id)
    return out


def run_test(model_dict, df_test, X_test):
    max_k = 200
    ks = np.array(list(range(1, max_k)))
    maes = np.zeros(max_k - 1)
    no_metric = model_dict["no_metric"]
    X_train = model_dict["X_train"]
    Y_train = model_dict["Y_train"]
    Y_train = np.repeat(Y_train[:, None], X_test.shape[0], axis=1).T
    Qrepresentation = model_dict["Qrepresentation"]
    knn = model_dict["knn"]
    print("Calculating distance matrix.", flush=True)
    start = time.perf_counter()
    if no_metric:
        D = pairwise_distances(X_test, X_train, n_jobs=-1)
    elif Qrepresentation == "fchl19-kernel":
        knn = VPTreeKNN19(kernel_args={"sigma": [1.0]})
        knn.fit(X_train, Y_train)
        D, neighbors = knn.kneighbours(X_test, n_neighbors=max_k)
        Y_train = Y_train[neighbors]
    #     elif Qrepresentation == "fchl-kernel-norm":
    #         knn = VPTreeKNN19(kernel_args={"sigma": [1.0]}, normalise_distance=True)
    #         knn.fit(X_train, Y_train)
    #         D, neighbors = knn.kneighbours(X_test, n_neighbors=max_k)
    #         Y_train = Y_train[neighbors]
    else:
        mlkr = model_dict["mlkr"]
        D = pairwise_distances(X_test, X_train, metric=mlkr.get_metric(), n_jobs=-1)
    sorted_indices = np.argsort(D, axis=1)
    # presort distance matrix
    D = np.take_along_axis(D, sorted_indices, axis=1)
    Y_sorted = np.take_along_axis(Y_train, sorted_indices, axis=1)
    print(
        f"Distance matrix calculation done and arrays sorted. Took {time.perf_counter() - start:.1f} s.",
        flush=True,
    )
    Y_true = df_test["extra"]["Y_true"].values
    for i, k in enumerate(ks):
        if knn.weights == "uniform":
            yhat = np.mean(Y_sorted[:, :k], axis=1)
        else:
            w = 1 / D[:, :k]
            yhat = np.average(Y_sorted[:, :k], axis=1, weights=w)
        yhat *= HARTREE_TO_KCALM
        maes[i] = np.mean(np.abs(Y_true - yhat))

    # Kanagawa2024: LOO-CV loss for knn = ((k+1)/k)^2 * 1/n * sum(loss(y_true, knn(k+1, x)))
    loocv = np.square((ks[1:] + 1) / ks[1:]) * maes[1:]
    # we do not get a value for the last element for obvious reasons
    loocv = np.append(loocv, [-1.0])
    # create the result dataframe
    df_out = pd.DataFrame({"k": ks, "MAE": maes})
    df_out.loc[:, "representation"] = Qrepresentation
    df_out.loc[:, "method"] = model_dict["method"]
    df_out.loc[:, "n_train"] = int(model_dict["sample_size"])
    df_out.loc[:, "job_id"] = int(model_dict["job_id"])
    df_out.loc[:, "loocv"] = loocv
    return df_out


if __name__ == "__main__":
    source_dir = Path(sys.argv[1])
    result_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        representations = sys.argv[3:]
        if not isinstance(representations, list):
            representations = [representations]
    else:
        representations = [
            "fchl",
            "mbdf",
            "fchl-kernel",
            "fchl-kernel-norm",
            "fchl-nometric",
            "mbdf-nometric",
        ]
    result_dir.mkdir(parents=True, exist_ok=True)

    cv_folds = 5
    fnames = [f.name for f in source_dir.glob("*knn*")]
    job_fnames = {}
    for job_id in range(5):
        fold_fnames = {}
        test_needle = rf"^(?:.*?_)?(knn)_{job_id}_fchl_1000_trainout\.pkl$"
        pattern = re.compile(test_needle)
        test_name = list(filter(pattern.match, fnames))
        if not test_name:
            raise FileNotFoundError(
                f"Could not find a test name among the filenames: {fnames}."
            )
        fold_fnames["test_path"] = test_name[0]
        for representation in representations:
            model_needle = rf"^(?:.*?_)?(knn)_{job_id}_{representation}_(\d+)\.pkl$"
            pattern = re.compile(model_needle)
            model_names = list(filter(pattern.match, fnames))
            fold_fnames[representation] = model_names
        job_fnames[job_id] = fold_fnames
    for job_id, fold_dict in job_fnames.items():
        df_test = load_test_data(source_dir / fold_dict["test_path"])
        print(
            f"Loaded test data from {source_dir / fold_dict['test_path']}.", flush=True
        )
        for representation, fnames in fold_dict.items():
            if representation == "test_path":
                continue
            out_path = result_dir / f"k_scaling_{representation}_{job_id}.pkl"
            if out_path.exists():
                df_representation = pd.read_pickle(out_path)
            else:
                df_representation = pd.DataFrame()
            print(f"Starting tests with {representation}.", flush=True)
            print("Calculating test representation.", flush=True)
            if "QM9" in str(source_dir):
                repr_hyperparams = {"max_atoms": 29, "elements": [1, 6, 7, 8, 9]}
            else:
                repr_hyperparams = {}
            X_test = calculate_representation(
                representation, df_test["xyz"]["structure"].values, **repr_hyperparams
            )
            for f in fnames:
                print(f"Working on {f}.")
                model_dict = load_model(source_dir / f)
                if (
                    not df_representation.empty
                    and (
                        (df_representation["representation"] == representation)
                        & (df_representation["n_train"] == model_dict["sample_size"])
                        & (df_representation["job_id"] == job_id)
                    ).any()
                ):
                    print(
                        f"Found results in {str(source_dir / f)}, not recalculating!",
                        flush=True,
                    )
                    continue
                if "fchl-kernel" in X_test:
                    X_train = model_dict["X_train"]
                    X_test, X_train = correct_fchl18_kernel_size(X_test, X_train)
                try:
                    df_res = run_test(model_dict, df_test, X_test)
                    df_representation = pd.concat(
                        (df_representation, df_res), ignore_index=True
                    )
                except Exception as e:
                    print(f"Could not perform test for {f}: {e}.", flush=True)
                with open(out_path, "wb") as pf:
                    df_representation.to_pickle(pf)
                print(f"Saved intermediate results to {out_path}", flush=True)
