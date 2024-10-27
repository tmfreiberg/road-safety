from typing import Type, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

from saaq_data_dictionary import shorthand

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib import rc

from utils import display, save_df


class distribution:
    @staticmethod
    def table(
        df: pd.DataFrame, target: str, dropna: bool = False, Print: bool = True
    ) -> pd.DataFrame:
        series: pd.Series = (
            df[target].value_counts(normalize=True, dropna=dropna).copy(deep=True)
        )
        series *= 100
        series.rename_axis(None, inplace=True)
        series.rename("%", inplace=True)

        output_df: pd.DataFrame = pd.DataFrame(series)
        output_df.columns.name: str = target
        try:
            output_df.index = output_df.index.map(shorthand[target])
        except:
            pass

        if Print:
            try:
                display(output_df)
            except:
                print(output_df)

        return output_df

    @staticmethod
    def plot(
        df: pd.DataFrame,
        target: str,
        dropna: bool = False,
        xticks_rotation: Union[int, None] = None,
        save_as: Union[Path, None] = None,
    ) -> None:
        series: pd.Series = (
            df[target].value_counts(normalize=True, dropna=dropna).copy(deep=True)
        )
        series *= 100
        try:
            series.index = series.index.map(shorthand[target])
        except:
            pass

        plt.title(f"Distribution: {target}")
        plt.grid(True, alpha=0.5, zorder=0)
        plt.bar(
            series.index.astype(str),
            series.values,
            color="darkblue",
            alpha=0.8,
            zorder=2,
        )
        plt.xlabel(target)

        if xticks_rotation is None:
            max_xtick_label_length = max(len(str(label)) for label in series.index)
            if max_xtick_label_length < 4:
                xticks_rotation = 0
            elif max_xtick_label_length > 8:
                xticks_rotation = 90
            else:
                xticks_rotation = 45

        plt.xticks(rotation=xticks_rotation)
        plt.ylabel("% accidents")
        if save_as is not None:
            plt.savefig(save_as)
        plt.show()

    @staticmethod
    def tuple_table(
        df: pd.DataFrame, targets: list, dropna: bool = False, Print: bool = True
    ) -> pd.DataFrame:
        if len(targets) == 1:
            return distribution.table(df, targets[0], dropna, Print)

        working_df = df[targets].copy(deep=True)
        for target in targets:
            try:
                working_df[target] = working_df[target].map(shorthand[target])
            except:
                pass

        tuple_series = 100 * working_df.apply(tuple, axis=1).value_counts(
            normalize=True, dropna=dropna
        )
        tuple_series.rename_axis(None, inplace=True)
        tuple_series.rename("%", inplace=True)

        output_df = pd.DataFrame(tuple_series)
        output_df.columns.name = "(" + ", ".join(targets) + ")"

        if Print:
            try:
                display(output_df)
            except:
                print(output_df)

        return output_df

    @staticmethod
    def tuple_plot(
        df: pd.DataFrame,
        targets: list,
        dropna: bool = False,
        xticks_rotation: Union[int, None] = None,
        save_as: Union[Path, None] = None,
    ) -> pd.DataFrame:

        working_df = df[targets].copy(deep=True)
        for target in targets:
            try:
                working_df[target] = working_df[target].map(shorthand[target])
            except:
                pass

        tuple_series = 100 * working_df.apply(tuple, axis=1).value_counts(
            normalize=True, dropna=dropna
        )
        tuple_series.rename_axis(None, inplace=True)
        tuple_series.rename("%", inplace=True)

        output_df = pd.DataFrame(tuple_series)
        output_df.columns.name = "(" + ", ".join(targets) + ")"

        plt.title(f"Distribution: {output_df.columns.name}")
        plt.grid(True, alpha=0.5, zorder=0)
        plt.bar(
            list(
                map(
                    lambda x: "(" + str(x[0]) + ", " + str(x[1]) + ")",
                    tuple_series.index,
                )
            ),
            tuple_series.values,
            color="darkblue",
            alpha=0.8,
            zorder=2,
        )
        plt.xlabel(output_df.columns.name)

        if xticks_rotation is None:
            xticks_labels, xticks_locs = plt.xticks()
            max_xtick_label_length = max(
                len(str(label)) for label in tuple_series.index
            )
            if max_xtick_label_length < 4:
                xticks_rotation = 0
            elif max_xtick_label_length > 8:
                xticks_rotation = 90
            else:
                xticks_rotation = 45

        plt.xticks(rotation=xticks_rotation)
        plt.ylabel("% accidents")
        if save_as is not None:
            plt.savefig(save_as)
        plt.show()


def custom_crosstab(
    col1: pd.Series,
    col2: pd.Series,
    dropna: bool = True,
    save_as: Union[Path, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not dropna:
        # Replace NaN values with a placeholder before computing crosstabs
        col1 = col1.fillna("NaN")
        col2 = col2.fillna("NaN")

    # Standard crosstabs with margins
    col1_by_col2 = pd.crosstab(col1, col2, dropna=dropna, margins=True)
    col1_by_col2_norm = (
        pd.crosstab(col1, col2, normalize="columns", dropna=dropna, margins=True)
        .mul(100)
        .round(2)
    )
    col1_by_col2_norm_rows = (
        pd.crosstab(col1, col2, normalize="index", dropna=dropna, margins=True)
        .mul(100)
        .round(2)
    )

    # The normalized crosstab will not contain column sums.
    # Even each column sums to 100%, we still want this:
    column_sums = pd.DataFrame(
        col1_by_col2_norm.sum(axis=0).round().astype(int), columns=["All"]
    ).T
    col1_by_col2_with_sum = pd.concat([col1_by_col2_norm, column_sums])

    row_sums = pd.DataFrame(
        col1_by_col2_norm_rows.sum(axis=1).round().astype(int), columns=["All"]
    )
    col1_by_col2_with_sum_rows = pd.concat([col1_by_col2_norm_rows, row_sums], axis=1)

    # Now we merge the two crosstabs, displaying (a,b) in each cell, where a is an integer and b is a percentage.
    combined_df = pd.merge(
        col1_by_col2, col1_by_col2_with_sum, left_index=True, right_index=True
    )
    for column in combined_df.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            combined_df[column_name] = list(
                zip(combined_df[column_name + "_x"], combined_df[column_name + "_y"])
            )

    combined_df_rows = pd.merge(
        col1_by_col2, col1_by_col2_with_sum_rows, left_index=True, right_index=True
    )
    for column in combined_df_rows.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            combined_df_rows[column_name] = list(
                zip(
                    combined_df_rows[column_name + "_x"],
                    combined_df_rows[column_name + "_y"],
                )
            )

    filtered_combined_df = combined_df.filter(regex="^(?!.*(_x|_y)$)")
    filtered_combined_df_rows = combined_df_rows.filter(regex="^(?!.*(_x|_y)$)")

    # Finally, we format so that (a,b) is displayed as a (b%)
    def format_tuple(a, b):
        return f"{a} (↓{b}%)"

    formatted_df = filtered_combined_df.map(
        lambda x: format_tuple(*x) if isinstance(x, tuple) else x
    )

    def format_tuple_rows(a, b):
        return f"{a} (→{b}%)"

    formatted_df_rows = filtered_combined_df_rows.map(
        lambda x: format_tuple_rows(*x) if isinstance(x, tuple) else x
    )

    if save_as is not None:
        save_df(col1_by_col2, save_as=save_as)
        save_df(col1_by_col2_norm, save_as=save_as, tack_on="norm")
        save_df(formatted_df, save_as=save_as, tack_on="formatted")
    # Depending on the situation, we might want numbers, percentages, or the combined table:
    return (
        col1_by_col2,
        col1_by_col2_norm,
        formatted_df,
        col1_by_col2_norm_rows,
        formatted_df_rows,
    )


def crosstab_plot(
    AxB: pd.DataFrame,
    ylabel: str = "% or number of all records",
    show: bool = True,
    save_as: Union[Path, None] = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    # Define color map for bars
    colors = plt.cm.get_cmap("inferno", len(AxB.columns)).reversed()

    # Calculate x positions for each bar group
    bar_width = 0.15
    bar_width_buffer = 0.05
    offset = (len(AxB.columns) - 1) * bar_width / 2
    bar_groups = len(AxB.columns) - 1
    xtickpos = (
        (bar_width_buffer + bar_width) * len(AxB.columns) * np.arange(len(AxB.index))
    )

    for i, col in enumerate(AxB.columns):
        x_pos = xtickpos + i * bar_width
        bar_positions = [
            x + offset - bar_groups * bar_width / 2 + i * bar_width for x in x_pos
        ]
        ax.bar(
            x_pos,
            AxB[col],
            width=bar_width,
            label=col,
            color=colors(i),
            alpha=1,
            zorder=4,
            align="center",
        )

    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_xlabel(f"{AxB.columns.name}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{AxB.columns.name} by {AxB.index.name}")

    ax.set_xticks(xtickpos + bar_width * (len(AxB.columns) - 1) / 2)
    ax.set_xticklabels(AxB.index, rotation=45)
    ax.legend()

    if save_as is not None:
        fig.savefig(save_as)

    if show:
        plt.show()

    return fig


def crosstab_animation(
    AxB: pd.DataFrame,
    ylabel: str = "% or number of all records",
    save_as: Union[Path, None] = None,
) -> FuncAnimation:
    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    y_min = 0
    y_max = 100

    # Define a colormap
    cmap = plt.cm.get_cmap("inferno", len(AxB.index)).reversed()

    # Create a ListedColormap based on the colormap
    colors = ListedColormap([cmap(i) for i in range(len(AxB.index))])

    # Define colors for each bar based on AxB.index
    colors_dict = {index: colors(i) for i, index in enumerate(AxB.index)}

    def update(frame):
        # Clear the previous plot
        ax.clear()
        # Plot the bar chart for the current column (frame)
        col = AxB.columns[frame]
        color = [colors_dict[index] for index in AxB.index]
        ax.bar(AxB.index, AxB[col], color=color, alpha=1, zorder=4)

        ax.grid(True, alpha=0.5, zorder=0)

        # Set labels and title
        ax.set_xlabel(f"{AxB.index.name}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{AxB.columns.name}: {col}")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)

        # Return the modified artists
        return (ax,)

    # Create the animation and assign it to a variable
    # rcParams setting fixes the problem of saved animated .gif being a sequence of frames where the plot titles and axis labels just get displayed on top of each other without clearing.
    plt.rcParams["savefig.facecolor"] = "white"
    anim = FuncAnimation(
        fig, update, frames=len(AxB.columns), interval=1000, repeat=True
    )
    # Save it (if applicable)
    if save_as is not None:
        try:
            anim.save(save_as, writer="pillow")
        except Exception as e:
            print(f"Unable to save animation: {e}")
    return anim
