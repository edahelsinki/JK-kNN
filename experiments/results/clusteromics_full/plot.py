from matplotlib.ticker import NullLocator, LogFormatterSciNotation, FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_error_and_times(
    df,
    error_label="MAE (kcal/mol)",
    chemical_accuracy=True,
    error_scaling=1.0,
    filter_extra=True,
):
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
    repr_map["FCHL-NOMETRIC"] = "FCHL19"
    repr_map["FCHL-KERNEL"] = "FCHL18"
    repr_map["MBDF-NOMETRIC"] = "MBDF"
    df_error.loc[:, "representation"] = df_error["representation"].map(repr_map)
    df_error.loc[df_error["ml_method"] == "KRR19", "ml_method"] = "KRR"
    df_error.loc[df_error["ml_method"] == "KRR", "metric"] = "KRR"
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
    repr_map["FCHL-KERNEL"] = "FCHL18"
    repr_map["MBDF-NOMETRIC"] = "MBDF"
    df_time.loc[:, "representation"] = df_time["representation"].map(repr_map)
    df_time.loc[df_time["ml_method"] == "KRR19", "ml_method"] = "KRR"
    df_time.loc[df_time["ml_method"] == "KRR", "metric"] = "KRR"
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
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
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
    ax = axs[0]
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
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # Increase font size of x-axis tick labels
    ax.tick_params(axis="x", labelsize=14)  # Change 14 to your preferred size
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x))))
    ax.tick_params(axis="y", labelsize=14)  # Change 14 to your preferred size
    ax = axs[1]
    for (method, rep, metric, identifier), gdf in df_time.loc[
        df_time["variable"] == "train_cpu"
    ].groupby(["ml_method", "representation", "metric", "identifier"]):
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
    ax.set_ylabel("Train CPU time (s)", size=14)
    # Set x-axis tick formatter to scientific notation
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # Increase font size of x-axis tick labels
    ax.tick_params(axis="x", labelsize=14)  # Change 14 to your preferred size
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=True))
    ax.tick_params(axis="y", labelsize=14)  # Change 14 to your preferred size

    ax = axs[2]
    for (method, rep, metric, identifier), gdf in df_time.loc[
        df_time["variable"] == "test_cpu"
    ].groupby(["ml_method", "representation", "metric", "identifier"]):
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
    ax.set_ylabel("Test CPU time (s)", size=14)
    # Set x-axis tick formatter to scientific notation
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # Increase font size of x-axis tick labels
    ax.tick_params(axis="x", labelsize=14)  # Change 14 to your preferred size
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    # ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=True))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelsize=14)  # Change 14 to your preferred size
    if chemical_accuracy:
        axs[0].axhline(1.0, color="k", label="Chemical accuracy")
    ax_handles, ax_labels = axs[0].get_legend_handles_labels()
    axs[0].legend(ax_handles, ax_labels, bbox_to_anchor=(-0.2, 1.0))
    fig.tight_layout()


print("Start plotting learning curves.")
path = Path(__file__).parent
df = pd.read_pickle(path / "learning_curve.pkl")
plot_error_and_times(df)
plt.savefig(path / "raw_clusteromics.pdf", dpi=300)
print(f"Saved learning curves to {path / 'raw_clusteromics.pdf'}")
