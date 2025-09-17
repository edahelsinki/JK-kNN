import sys
from matplotlib.ticker import NullLocator, LogFormatterSciNotation, FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import pickle
from ase.visualize.plot import plot_atoms
from ase import Atoms
from typing import Iterable
from collections import defaultdict

path = Path(__file__).parent
sys.path.append(str(path.parent.parent))
from constants import *
from paths import JKML_PATH

sys.path.append(str(JKML_PATH))
from src.data import substract_monomers


path = Path(__file__).parent

# Plot learning curves
print("Start plotting learning curves.")
width: float = 1.0
aspect: float = 1.0
cols: int = 1
rows: int = 1
page_width: float = 347.0
if width == 1.0:
    width = 0.99
scale = page_width / 72.27  # from points to inches
size = (width * scale, width / cols * rows / aspect * scale)
sns.set_theme(
    context={k: v for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    # palette="bright",
    # font="cmr10",
    rc={
        "figure.figsize": size,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 1e-4,
    },
)
df_direct = pd.read_pickle(path / "direct_learning_curve.pkl")
df_delta = pd.read_pickle(path / "delta_learning_curve.pkl")
filter_extra = True
chemical_accuracy = True
error_scaling = 1.0
fig = plt.figure(constrained_layout=True)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for row_id, df in enumerate([df_direct, df_delta]):
    df_error = df.melt(
        id_vars=["ml_method", "job", "representation", "n", "identifier"],
        value_vars=["error"],
    )
    df_error["value"] *= error_scaling
    df_error.loc[:, "metric"] = "MLKR"
    df_error.loc[
        df_error["representation"].isin(["FCHL-KERNEL", "FCHL19-KERNEL"]), "metric"
    ] = "Kernel"
    df_error.loc[df_error["representation"].str.contains("-NOMETRIC"), "metric"] = (
        "Euclidean"
    )
    df_error.loc[:, "repr_raw"] = df_error["representation"]
    repr_map = {k: k for k in df_error["representation"].unique()}
    repr_map["FCHL-KERNEL"] = "FCHL18"
    repr_map["FCHL19-KERNEL"] = "FCHL19"
    repr_map["FCHL-NOMETRIC"] = "FCHL19"
    repr_map["MBDF-NOMETRIC"] = "MBDF"
    df_error.loc[:, "representation"] = df_error["representation"].map(repr_map)
    df_error.loc[df_error["ml_method"] == "KRR19", "ml_method"] = "KRR"
    df_error.loc[df_error["ml_method"] == "KRR", "metric"] = "KRR"
    if filter_extra:
        df_error = df_error.loc[
            df_error["identifier"].isin(
                [
                    "KNN (FCHL19-KERNEL)",
                    "KNN (FCHL-NOMETRIC)",
                    "KNN (FCHL19)",
                    "MLKR (FCHL19)",
                    "KRR19 (FCHL19)",
                ]
            ),
            :,
        ]
    mark_dict = {"Euclidean": "o", "MLKR": "x", "Kernel": "D", "KRR": "*"}
    linestyle = {
        "KRR19 (FCHL19)": "solid",
        "MLKR (FCHL19)": "solid",
        "KNN (FCHL19)": "dashed",
        "KNN (FCHL-NOMETRIC)": "dotted",
        "KNN (FCHL19-KERNEL)": "dashdot",
    }
    palette = sns.color_palette("deep")
    if filter_extra:
        colors = dict(zip(sorted(df_error["identifier"].unique()), palette))
        colors = {
            "KRR19 (FCHL19)": "C2",
            "MLKR (FCHL19)": "C1",
            "KNN (FCHL19)": sns.color_palette("rocket")[0],
            "KNN (FCHL-NOMETRIC)": sns.color_palette("rocket")[1],
            "KNN (FCHL19-KERNEL)": sns.color_palette("rocket")[2],
        }
    else:
        colors = dict(zip(sorted(df_error["representation"].unique()), palette))
    # error plot
    ax = axs[row_id]
    for (method, rep, metric, identifier), gdf in df_error.groupby(
        ["ml_method", "representation", "metric", "identifier"]
    ):
        gdf = gdf.sort_values("n")
        meandf = gdf.groupby("n").mean(numeric_only=True).reset_index()
        maxdf = gdf.groupby("n").max(numeric_only=True).reset_index()
        mindf = gdf.groupby("n").min(numeric_only=True).reset_index()
        ax.plot(
            meandf["n"],
            meandf["value"],
            label=f"{method} | {metric}" if method == "KNN" else f"{method}",
            color=colors[rep] if not filter_extra else colors[identifier],
            linestyle=linestyle[identifier],
            marker=mark_dict[metric],
        )
        ax.fill_between(
            meandf["n"],
            mindf["value"],
            maxdf["value"],
            color=colors[rep] if not filter_extra else colors[identifier],
            alpha=0.2,
        )
    ax.set_xscale("log")
    ax.axhline(1.0, c="k")
    ax.set_ylabel("Test set MAE (kcal/mol)", size=14)
    ax.set_xlabel("Train set size", size=14)
    # Set x-axis tick formatter to scientific notation
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # Increase font size of x-axis tick labels
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis="x", labelsize=14)  # Change 14 to your preferred size
    ax.set_yticklabels(ax.get_yticks(), size=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
    if chemical_accuracy:
        ax.axhline(1.0, color="k", label="Chemical accuracy")
    if row_id == 0:
        ax.set_title("Direct learning", size=14)
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        ax.legend()
    else:
        ax.set_title(r"$\Delta$-learning", size=14)
    fig.tight_layout()
plt.savefig(path / "raw_sa-w_combined_error.pdf", dpi=300)
print(f"Saved learning curves to {path / 'raw_sa-w_combined_error.pdf'}")

# Plot times
print("Start plotting time curves.")
fig = plt.figure(constrained_layout=True)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for row_id, df in enumerate([df_direct, df_direct]):
    df_time = df.melt(
        id_vars=["ml_method", "job", "representation", "n", "identifier"],
        value_vars=["train_cpu", "train_wall", "test_cpu", "test_wall"],
    )
    df_time = df_time.drop_duplicates()
    df_time.loc[:, "metric"] = "MLKR"
    df_time.loc[
        df_time["representation"].isin(["FCHL-KERNEL", "FCHL19-KERNEL"]), "metric"
    ] = "Kernel"
    df_time.loc[df_time["representation"].str.contains("-NOMETRIC"), "metric"] = (
        "Euclidean"
    )
    df_time.loc[:, "repr_raw"] = df_time["representation"]
    repr_map = {k: k for k in df_time["representation"].unique()}
    repr_map["FCHL-KERNEL"] = "FCHL18"
    repr_map["FCHL-NOMETRIC"] = "FCHL19"
    repr_map["FCHL19-KERNEL"] = "FCHL19"
    repr_map["MBDF-NOMETRIC"] = "MBDF"
    df_time.loc[:, "representation"] = df_time["representation"].map(repr_map)
    df_time.loc[df_time["ml_method"] == "KRR19", "ml_method"] = "KRR"
    df_time.loc[df_time["ml_method"] == "KRR", "metric"] = "KRR"
    if filter_extra:
        df_time = df_time.loc[
            df_time["identifier"].isin(
                [
                    "KNN (FCHL19-KERNEL)",
                    "KNN (FCHL-NOMETRIC)",
                    "KNN (FCHL19)",
                    "MLKR (FCHL19)",
                    "KRR19 (FCHL19)",
                ]
            ),
            :,
        ]
    mark_dict = {"Euclidean": "o", "MLKR": "x", "Kernel": "D", "KRR": "*"}
    linestyle = {
        "KRR19 (FCHL19)": "solid",
        "MLKR (FCHL19)": "solid",
        "KNN (FCHL19)": "dashed",
        "KNN (FCHL-NOMETRIC)": "dotted",
        "KNN (FCHL19-KERNEL)": "dashdot",
    }
    palette = sns.color_palette("deep")
    if filter_extra:
        colors = dict(zip(sorted(df_time["identifier"].unique()), palette))
        colors = {
            "KRR19 (FCHL19)": "C2",
            "MLKR (FCHL19)": "C1",
            "KNN (FCHL19)": sns.color_palette("rocket")[0],
            "KNN (FCHL-NOMETRIC)": sns.color_palette("rocket")[1],
            "KNN (FCHL19-KERNEL)": sns.color_palette("rocket")[2],
        }
    else:
        colors = dict(zip(sorted(df_time["representation"].unique()), palette))
    # error plot
    ax = axs[row_id]
    for (method, rep, metric, identifier), gdf in df_time.groupby(
        ["ml_method", "representation", "metric", "identifier"]
    ):
        gdf = (
            gdf.loc[gdf["variable"] == "train_cpu"]
            if row_id == 0
            else gdf.loc[gdf["variable"] == "test_cpu"]
        )
        gdf = gdf.sort_values("n")
        meandf = gdf.groupby("n").mean(numeric_only=True).reset_index()
        maxdf = gdf.groupby("n").max(numeric_only=True).reset_index()
        mindf = gdf.groupby("n").min(numeric_only=True).reset_index()
        ax.plot(
            meandf["n"],
            meandf["value"],
            label=f"{method} | {metric}" if method == "KNN" else f"{method}",
            color=colors[rep] if not filter_extra else colors[identifier],
            linestyle=linestyle[identifier],
            marker=mark_dict[metric],
        )
        ax.fill_between(
            meandf["n"],
            mindf["value"],
            maxdf["value"],
            color=colors[rep] if not filter_extra else colors[identifier],
            alpha=0.2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Train set size", size=14)
    # Set x-axis tick formatter to scientific notation
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # Increase font size of x-axis tick labels
    ax.tick_params(axis="x", labelsize=14)  # Change 14 to your preferred size
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.tick_params(axis="y", labelsize=14)  # Change 14 to your preferred size
    ax.set_title("Direct learning", size=14)
    if row_id == 0:
        ax.set_ylabel("Train CPU time (s)", size=14)
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        ax.legend()
    else:
        ax.set_ylabel("Test CPU time (s)", size=14)
    fig.tight_layout()
plt.savefig(path / "raw_sa-w_combined_time.pdf", dpi=300)
print(f"Saved time curves to {path / 'raw_sa-w_combined_time.pdf'}")

# Plot k-scaling
print("Start plotting k sensitivity curves.")
fnames = [f.name for f in path.glob("k_scaling_fchl*")]
df_master = pd.DataFrame()
for f in fnames:
    if f == "k_scaling_fchl.pkl":
        continue
    with open(path / f, "rb") as dbf:
        dbf.seek(0)
        df = pd.read_pickle(dbf)
        if "-nometric" in f:
            df.loc[:, "representation"] = df["representation"] + "-nometric"
        df_master = pd.concat((df_master, df), ignore_index=True)
df_master = df_master.sort_values("representation")
df_plot = df_master.loc[
    (df_master["representation"].isin(["fchl", "fchl-nometric"]))
    & (df_master["k"] <= 30)
]
palette = sns.color_palette("flare", n_colors=len(df_plot["n_train"].unique()))
colors = dict(zip(sorted(df_plot["n_train"].unique().astype(int)), palette))
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(12, 5))
for rep, rdf in df_plot.groupby("representation"):
    if rep == "fchl":
        ax = axs[0]
        title = "KNN | FCHL19 | MLKR"
    else:
        ax = axs[1]
        title = "KNN | FCHL19 | Euclidean"
    for n, ndf in rdf.groupby("n_train"):
        ndf = ndf.sort_values("k")
        meandf = ndf.groupby("k").mean(numeric_only=True).reset_index()
        maxdf = ndf.groupby("k").max(numeric_only=True).reset_index()
        mindf = ndf.groupby("k").min(numeric_only=True).reset_index()
        ax.plot(meandf["k"], meandf["MAE"], color=colors[n], label=str(int(n)))
        ax.fill_between(
            meandf["k"], mindf["MAE"], maxdf["MAE"], color=colors[n], alpha=0.2
        )
        ax.set_title(title)
axs[1].legend(
    title="Train set size", fontsize=14, title_fontsize=15, bbox_to_anchor=(1.01, 0.8)
)
axs[0].set_ylabel("Test set MAE (kcal/mol)", size=14)
axs[0].set_yticks(axs[0].get_yticks())
axs[0].set_yticklabels(axs[0].get_yticks(), size=14)
axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
for ax in axs:
    # dummy set ticks to get rid of plt warning
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    # increase font size
    ax.set_xticklabels(ax.get_xticks(), size=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
fig.text(
    0.5, -0.02, "Number of nearest neighbours", ha="center", fontdict={"fontsize": 14}
)
fig.tight_layout()
plt.savefig(path / "saw_delta_k_scaling.pdf", dpi=300, bbox_inches="tight")
print(f"Saved k sensitivity curves to {path / 'saw_delta_k_scaling.pdf'}")

# recreate UE plots
print("Start plotting uncertainty analysis curves.")
unc = pd.read_pickle(path / "uncertainty_fchl.pkl")
unc = unc.loc[unc["n_train"] == 13000]
unc = unc.loc[unc["job_id"] == 0]
unc_p = unc.sort_values("Y_true")
fig = plt.figure(figsize=(13, 6), constrained_layout=True, edgecolor="k")
gs = fig.add_gridspec(nrows=1, ncols=2)
# example_indices = [30, 60, 90]
example_indices = [150, 300, 450]
ax_left = fig.add_subplot(gs[0, 0])
y = unc_p.loc[unc["n_train"] == 13000]["Y_true"].values
yhat = unc_p.loc[unc["n_train"] == 13000]["Y_pred"].values
q25 = yhat - unc_p.loc[unc["n_train"] == 13000]["q25"].values
q75 = unc_p.loc[unc["n_train"] == 13000]["q75"].values - yhat
exact_qs = []
empirical_qs = []
cols = list(unc.columns)
pattern = re.compile(r"^q\d{2}$")
quantiles = [s[1:] for s in cols if pattern.fullmatch(s)]
for q in quantiles:
    exact_qs.append(float(q))
    empirical_qs.append(np.sum(unc_p["Y_true"] < unc_p[f"q{q}"]) / len(unc_p.index))
exact_qs = np.array(exact_qs)
empirical_qs = np.array(empirical_qs)
within_50 = np.logical_and(yhat - q25 < y, y < q75 + yhat)
error = np.abs(y - yhat)
qerr = np.array([q25, q75])
num_examples = 10
e_vals = np.linspace(min(y), max(y), num=num_examples + 2)[1 : num_examples + 1]
example_indices = [np.argmin(np.abs(y - ev)) for ev in e_vals]
knn_examples = [example_indices[2], example_indices[4], example_indices[8]]
vmin = np.min(error)
vmax = np.max(error)
sc = ax_left.scatter(
    y, yhat - y, c=error, cmap="plasma", s=3, alpha=0.1, vmin=vmin, vmax=vmax
)
sc = ax_left.scatter(
    y[example_indices],
    yhat[example_indices] - y[example_indices],
    c=error[example_indices],
    cmap="plasma",
    s=30,
    vmin=vmin,
    vmax=vmax,
)
ax_left.errorbar(
    y[example_indices],
    yhat[example_indices] - y[example_indices],
    yerr=qerr[:, example_indices],
    linestyle="none",
    ecolor="k",
    capsize=5,
)
ax_left.plot(y, y - y, c="r", linestyle="--")
ax_left.set_ylim([-20, 20])
ax_left.set_xlabel(r"$\Delta E_{true}$", fontdict={"size": 14})
ax_left.set_ylabel(r"$\Delta E_{true} - \Delta E_{pred}$", fontdict={"size": 14})
ax_left.tick_params(axis="x", labelsize=14)
ax_left.tick_params(axis="y", labelsize=14)
colors = plt.get_cmap("tab10")
letters = ["A", "B", "C"]
for ci, nni in enumerate(knn_examples):
    xoffset = 2.0
    yoffset = 1.0
    x0, y0 = y[nni] - xoffset, yhat[nni] - y[nni] - qerr[0, nni] - yoffset
    w = xoffset * 2
    h = (
        yhat[nni]
        - y[nni]
        + qerr[1, nni]
        - (yhat[nni] - y[nni] - qerr[0, nni])
        + 2 * yoffset
    )
    # add box
    ax_left.add_patch(
        plt.Rectangle((x0, y0), w, h, ec=colors(ci), fc="none", linestyle="--", lw=2)
    )
    ax_left.text(
        x0 - xoffset, y0 + h + yoffset * 2, letters[ci], fontsize=16, color=colors(ci)
    )
ax_left.grid(False)

ax_right = fig.add_subplot(gs[0, 1])
ax_right.scatter(exact_qs, [x * 100 for x in empirical_qs])
ax_right.plot(np.linspace(0, 100), np.linspace(0, 100), c="r", linestyle="--")
ax_right.tick_params(axis="x", labelsize=14)
ax_right.tick_params(axis="y", labelsize=14)
ax_right.set_xlabel("Theoretical quantiles", fontdict={"size": 14})
ax_right.set_ylabel("Empirical quantiles", fontdict={"size": 14})
fig.savefig(path / "uncertainty_base.pdf", dpi=150)
print(f"Saved uncertainty plots to {path / 'uncertainty_base.pdf'}")

# plot nearest neighbours
print("Start plotting nearest neighbours examples.")
no_metric = False
VARS_PKL = path / "saw_delta_knn_0_fchl_13000.pkl"

Qrepresentation = "fchl"
with open(VARS_PKL, "rb") as f:
    if no_metric:
        X_train, Y_train, X_atoms, knn_params, train_metadata = pickle.load(f)
    elif Qrepresentation == "fchl-kernel":
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
knn_params["metric"] = mlkr.get_metric()
knn = KNeighborsRegressor(**knn_params)
knn.fit(X_train, Y_train)
print("\tLoaded k-nn model.")


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


def calculate_representation(Qrepresentation, strs, **repr_kwargs):
    if Qrepresentation == "fchl":
        return _generate_fchl19(strs, **repr_kwargs)
    else:
        raise NotImplementedError(
            f"Representation 'f{Qrepresentation}' not supported with the k-NN model!"
        )


# Load test data
Qrepresentation = "fchl"
TEST_PATH = path / "saw_delta_knn_0_fchl_13000_trainout.pkl"
with open(TEST_PATH, "rb") as f:
    f.seek(0)
    df_test = pd.read_pickle(f)
df_test.head(10)
# also calculate the representations
X_test = calculate_representation(Qrepresentation, df_test["xyz"]["structure"].values)
# load fold train data
TRAIN_PATH = path / "db_high_0_13000.pkl"
with open(TRAIN_PATH, "rb") as f:
    f.seek(0)
    df_train = pd.read_pickle(f)
df_train.head(10)
Xdf = calculate_representation(Qrepresentation, df_train["xyz"]["structure"].values)
print("\tLoaded raw train and test data.")

plot_dict = {
    l: {"y_pred": [], "structures": [], "error": -1, "y_true": -1} for l in letters
}
for l, ei in zip(letters, knn_examples):
    file_basename = unc_p["file_basename"].values[ei]
    test_index = df_test.loc[df_test[("info", "file_basename")] == file_basename].index[
        0
    ]
    plot_dict[l]["error"] = error[ei]
    plot_dict[l]["y_pred"].append(yhat[ei])
    plot_dict[l]["y_true"] = y[ei]
    test_str = df_test.loc[df_test[("info", "file_basename")] == file_basename]["xyz"][
        "structure"
    ].values[0]
    plot_dict[l]["structures"].append(test_str)
    test_repr = calculate_representation(Qrepresentation, [test_str])
    dist, index = knn.kneighbors(test_repr, n_neighbors=5)
    # match indices in df_train and X_train
    train_indices = []
    for ki in index.flatten():
        diff = Xdf - X_train[ki, :]
        min_idx = np.argmin(np.sum(np.abs(diff), axis=1))
        train_str = df_train["xyz"]["structure"].values[min_idx]
        assert np.allclose(
            X_train[ki, :], calculate_representation(Qrepresentation, [train_str])
        )
        plot_dict[l]["structures"].append(train_str)
        plot_dict[l]["y_pred"].append(Y_train[ki] * HARTREE_TO_KCALM)
print("\tProcessed data for plotting.")


def draw_structure_table(
    structures,
    y_pred,
    y_true,
    title_id="A",
    error_value=1.2,
    units="kcal/mol",
    figsize=(10, 5),
    yfmt="{:.2f}",
    row_hspace=0.01,  # ↓ smaller = tighter vertical spacing
    row_pad=0.01,
):  # ↓ padding
    """
    Render a 3x6 'table' of axes with:
      - Top title: 'Structure A (error = 1.2 kcal/mol)'
      - Row 1 (header): [A][nearest neighbours spanning 5 cols]
      - Row 2 (plots):  six square plots via plot_atoms(structures[i])
      - Row 3 (values): text y_true[i] under each plot

    Parameters
    ----------
    structures : list
        List-like of at least 6 structures.
    y_true : list or array
        List-like of at least 6 numeric values aligned with structures.
    plot_atoms : callable
        Function to draw a structure. Should accept (structure, ax=...).
    title_id : str
        The ID/letter for the first column header (e.g., 'A').
    error_value : float
        Value to show in the title.
    units : str
        Units to display in the title.
    figsize : tuple
        Figure size.
    yfmt : str
        Format string for y-values in the last row.
    """

    if len(structures) < 6 or len(y_pred) < 6:
        raise ValueError("Need at least 6 structures and 6 y-values.")

    structures6 = structures[:6]
    y6 = y_pred[:6]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(
        h_pad=row_pad,
        w_pad=0.02,  # pad in inches
        hspace=row_hspace,
        wspace=0.02,  # fractional spacing
    )

    gs = GridSpec(
        nrows=3,
        ncols=6,
        figure=fig,
        height_ratios=[0.18, 1.0, 0.22],
        width_ratios=[1, 1, 1, 1, 1, 1],
    )

    fig.suptitle(
        f"Structure {title_id} (error = {error_value:.2f} {units})", y=0.99, fontsize=14
    )

    # --- Row 1: Header ---
    # Col 0: letter A
    ax_header_left = fig.add_subplot(gs[0, 0])
    ax_header_left.set_axis_off()
    ax_header_left.text(0.5, 0.5, rf"{title_id}", ha="center", va="center", fontsize=13)
    ax_header_left.text(
        0.5,
        0.2,
        rf"$\Delta E_{{{'true'}}} = {y_true:.2f}$ {units}",
        ha="center",
        va="center",
        fontsize=13,
    )

    # Cols 1-5 merged: "nearest neighbours"
    ax_header_right = fig.add_subplot(gs[0, 1:6])
    ax_header_right.set_axis_off()
    ax_header_right.text(
        0.5, 0.5, "Nearest neighbours", ha="left", va="center", fontsize=13
    )

    # --- Row 2: Structure plots (square) ---
    axes_plots = []
    for col in range(6):
        ax = fig.add_subplot(gs[1, col])
        # Make each plotting panel square
        try:
            ax.set_box_aspect(1)  # Matplotlib >= 3.3
        except Exception:
            ax.set_aspect("equal", adjustable="box")
        # Hide ticks for a clean 'table' look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Draw the structure in this cell
        try:
            plot_atoms(structures6[col], ax=ax)
        except TypeError:
            # If plot_atoms doesn't accept ax, set current axes and call
            plot_atoms(structures6[col], ax=ax)
        axes_plots.append(ax)

    # --- Row 3: y-values under each plot ---
    for col in range(6):
        ax = fig.add_subplot(gs[2, col])
        ax.set_axis_off()
        # Centered text; adjust fontsize as you like
        y_text = (
            yfmt.format(y6[col]) if isinstance(y6[col], (int, float)) else str(y6[col])
        )
        ax.text(0.5, 0.5, y_text, ha="center", va="center", fontsize=11)

    return fig


for l in letters:
    fig = draw_structure_table(
        plot_dict[l]["structures"],
        plot_dict[l]["y_pred"],
        plot_dict[l]["y_true"],
        l,
        plot_dict[l]["error"],
        row_hspace=0.001,
        row_pad=0.001,
    )
    fig.savefig(path / f"nn5_raw_{l}.pdf", dpi=150)
    print(f"\tSaved NNs for {l} to {path / f'nn5_raw_{l}.pdf'}.")
print("Done.")
