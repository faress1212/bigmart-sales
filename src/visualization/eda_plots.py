"""
Exploratory Data Analysis (EDA) plotting utilities for the Big Mart Sales
Visualization project.

Covers line plots, bar charts, histograms, box/violin plots, scatter plots,
and bubble-style scatter plots using Seaborn.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR


def _save_and_show(save_path: str = None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_line(df: pd.DataFrame, x: str, y: str, n_rows: int = 50, save_path: str = None) -> None:
    """
    Line plot of two numeric columns for the first n_rows of the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        n_rows (int): Number of rows to plot (avoids overplotting on large data).
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=x, y=y, data=df[:n_rows])
    plt.title(f"{x} vs {y} (first {n_rows} rows)")
    _save_and_show(save_path)


def plot_bar(df: pd.DataFrame, x: str, y: str, save_path: str = None) -> None:
    """
    Bar chart comparing a numeric column across categories.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): Numeric column for bar length.
        y (str): Categorical column for bar grouping.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(x=x, y=y, data=df)
    plt.title(f"{x} by {y}")
    _save_and_show(save_path)


def plot_histogram(df: pd.DataFrame, column: str, kde: bool = False, save_path: str = None) -> None:
    """
    Histogram of a numeric column, with an optional KDE overlay.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to plot.
        kde (bool): Whether to overlay a kernel density estimate.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(x=column, data=df, edgecolor="black", kde=kde)
    plt.title(f"{column} Distribution")
    _save_and_show(save_path)


def plot_boxplot(df: pd.DataFrame, column: str, save_path: str = None) -> None:
    """
    Boxplot of a numeric column to visualize spread and outliers.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to plot.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title(f"{column} Boxplot")
    _save_and_show(save_path)


def plot_violin(df: pd.DataFrame, column: str, save_path: str = None) -> None:
    """
    Violin plot of a numeric column, showing distribution shape and spread.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to plot.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=df[column])
    plt.title(f"{column} Violin Plot")
    _save_and_show(save_path)


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None, n_rows: int = 200, save_path: str = None) -> None:
    """
    Scatter plot of two numeric columns, optionally colored by a third column.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        hue (str, optional): Column name to color points by.
        n_rows (int): Number of rows to plot (avoids overplotting on large data).
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, data=df[:n_rows], hue=hue)
    title = f"{x} vs {y}" + (f" by {hue}" if hue else "")
    plt.title(title)
    _save_and_show(save_path)


def plot_bubble(df: pd.DataFrame, x: str, y: str, hue: str, col: str = None, style: str = None, n_rows: int = 200, save_path: str = None) -> None:
    """
    Bubble-style scatter plot (relplot) with color/style/facet encodings for
    an extra categorical dimension.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        hue (str): Column name to color points by.
        col (str, optional): Column name to facet into separate subplots.
        style (str, optional): Column name to vary marker style by.
        n_rows (int): Number of rows to plot (avoids overplotting on large data).
        save_path (str, optional): If provided, save the figure to this path.
    """
    g = sns.relplot(
        kind="scatter",
        x=x, y=y,
        data=df[:n_rows],
        hue=hue,
        col=col,
        style=style,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        g.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a correlation heatmap for all numeric columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    _save_and_show(save_path)


def generate_all_plots(df: pd.DataFrame, figures_dir: str = FIGURES_DIR) -> None:
    """
    Generate and save the full set of EDA plots in one call.

    Args:
        df (pd.DataFrame): Cleaned dataframe (before categorical encoding,
                            so category labels read naturally in the plots).
        figures_dir (str): Directory to save figures into.
    """
    os.makedirs(figures_dir, exist_ok=True)

    plot_histogram(df, "item_outlet_sales", kde=True, save_path=os.path.join(figures_dir, "sales_distribution.png"))
    plot_boxplot(df, "item_outlet_sales", save_path=os.path.join(figures_dir, "sales_boxplot.png"))
    plot_bar(df, "item_outlet_sales", "outlet_type", save_path=os.path.join(figures_dir, "sales_by_outlet_type.png"))
    plot_bar(df, "item_outlet_sales", "outlet_size", save_path=os.path.join(figures_dir, "sales_by_outlet_size.png"))
    plot_scatter(
        df, "item_mrp", "item_outlet_sales", hue="outlet_type",
        save_path=os.path.join(figures_dir, "mrp_vs_sales.png"),
    )
    plot_violin(df, "item_visibility", save_path=os.path.join(figures_dir, "visibility_violin.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(figures_dir, "correlation_heatmap.png"))
