from matplotlib.ticker import NullLocator, LogFormatterSciNotation, FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


path = Path(__file__).parent
df_extra = pd.read_pickle(path / "learning_curve.pkl")
df_delta = pd.read_pickle(path.parent / "SA-W/delta_learning_curve.pkl")
width: float = 1.0
aspect: float = 1.0
cols: int = 1
rows: int = 1
page_width: float = 347.0
if width == 1.0:
    width = 0.99
scale = page_width / 72.27  # from points to inches
size = (width * scale, width / cols * rows / aspect * scale)
filter_extra = True
chemical_accuracy = True
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
dfs = {}
for df, label in [(df_delta, "delta"), (df_extra, "extra")]:
    df_error = df.melt(
        id_vars=["ml_method", "job", "representation", "n", "identifier"],
        value_vars=["error"],
    )
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
                ["KNN (FCHL19)", "MLKR (FCHL19)", "KRR19 (FCHL19)"]
            ),
            :,
        ]
    dfs[label] = df_error
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
    for (method, rep, metric, identifier), gdf in df_error.groupby(
        ["ml_method", "representation", "metric", "identifier"]
    ):
        gdf = gdf.sort_values("n")
        meandf = gdf.groupby("n").mean(numeric_only=True).reset_index()
        maxdf = gdf.groupby("n").max(numeric_only=True).reset_index()
        mindf = gdf.groupby("n").min(numeric_only=True).reset_index()
        if label == "delta":
            ax.plot(
                meandf["n"],
                meandf["value"],
                label=f"{method} | {metric}" if method == "KNN" else f"{method}",
                color=colors[rep] if not filter_extra else colors[identifier],
                linestyle=linestyle[identifier],
                marker=mark_dict[metric],
                alpha=0.4,
            )
        else:
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
ax.legend(bbox_to_anchor=(1.05, 1.0))
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
ax_handles, ax_labels = ax.get_legend_handles_labels()
ax.legend(ax_handles, ax_labels, bbox_to_anchor=(-0.2, 1.0))
fig.tight_layout()
plt.savefig(path / "raw_saw_extrap.pdf", dpi=300)
