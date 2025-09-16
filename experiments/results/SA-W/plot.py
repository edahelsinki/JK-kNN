from matplotlib.ticker import NullLocator, LogFormatterSciNotation, FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

path = Path(__file__).parent

# Plot learning curves
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
        print(df_error["identifier"].unique())
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

# Plot times
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
        print(df_time["identifier"].unique())
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

# Plot k-scaling
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
axs[0].set_yticklabels(axs[0].get_yticks(), size=14)
axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
for ax in axs:
    ax.set_xticklabels(ax.get_xticks(), size=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
fig.text(
    0.5, -0.02, "Number of nearest neighbours", ha="center", fontdict={"fontsize": 14}
)
fig.tight_layout()
plt.savefig(path / "saw_delta_k_scaling.pdf", dpi=300, bbox_inches="tight")
