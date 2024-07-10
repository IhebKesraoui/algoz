#!/usr/bin/env python3

import logging
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.interval import IntervalIndex
import numpy as np
from numpy.linalg import LinAlgError

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from nanoz.utils import is_notebook
from nanoz.modeling import AvailableAlgorithm
from nanoz.evaluation import gaussian_kde


def _set_grid(ax, show_grid):
    """
    Configure the grid lines.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        Axis where to show the grid lines.
    show_grid : bool
        If is True, show the grid lines.
    """
    if show_grid:
        ax.grid(which='major', linestyle='-', alpha=0.9)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.7)


def _set_y_lim(ax, value):
    """
    Configure y-axis.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        Axis where to set the y-limits.
    value : int or list of int
        Set the y-limits of the axis.
        If is an integer, set the bottom value. If is a list set the bottom and top values.
    """
    if isinstance(value, list):
        ax.set_ylim(value)
    elif isinstance(value, int):
        ax.set_ylim(bottom=value)


def _set_x_scale(ax, value):
    """
    Set the x-axis' scale.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        Axis where to apply on the x-axis scale.
    value : {'linear', 'log', 'symlog','logit'}
        The axis scale type to apply on the x-axis.
    """
    ax.set_xscale(value)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())


def _set_x_ticks(ax, x_ticks, x_rotate=0):
    """
    Set the x-axis' tick locations and labels, optionally change the rotation of the x-axis labels .

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        Axis where to apply on the x-axis scale.
    x_ticks : array-like
        List of tick locations and labels of the x-axis.
    x_rotate: float or {'vertical', 'horizontal'}, default=0
        Rotation in degrees to apply at the x-axis ticks.
    """
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        if x_rotate == 0:
            ax.set_xticklabels([str(s).rstrip('0').rstrip('.') for s in x_ticks])
        else:
            ax.set_xticklabels([str(s).rstrip('0').rstrip('.') for s in x_ticks], rotation=x_rotate)


class NanozFig:
    """
    Figure template with Nanoz style.

    Attributes
    ----------
    fig_size : tuple of float
        Width and height of the figure in inches.
    dpi : float
        The resolution of the figure in dots-per-inch.
    face_color : matplotlib color
        The background color of the figure.
        For all color, see https://matplotlib.org/stable/gallery/color/named_colors.html.
    edge_color : matplotlib color
        The border color of the figure.
        For all color, see https://matplotlib.org/stable/gallery/color/named_colors.html.
    title_fontsize : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        Font size of the figure title.
        If a float, the font size in points. The string values denote sizes relative to the default font size.
    colors_line : list of matplotlib color
        List of format string from matplotlib, e.g. 'r' for red.
        For all color, see https://matplotlib.org/stable/gallery/color/named_colors.html.
    line_styles : list of matplotlib line style
        List of format string from matplotlib, e.g. '-' for solid line.
        For all line style, see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html.
    marker_styles : list of matplotlib marker style
        List of format string from matplotlib, e.g. 'o' for circle.
        For all marker style, see https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html.

    Methods
    -------
    plot_one_y_axis(y, x, title, y_label, x_label, x_scale, x_ticks, x_rotate, y_lim, show_grid, save_path)
        Plot y versus x as lines with matplotlib library.
    plot_double_y_axis(*y, x, title, y_label, x_label, x_scale, x_ticks, x_rotate, y_lim, y_lim_right,
                       show_grid, save_path)
        Comparison of plots y versus x as lines on two y-axis with matplotlib library.
    scatter_one_y_axis(y, x, title, y_label, x_label, x_scale, x_ticks, x_rotate, y_lim, show_grid, save_path)
        Plot y versus x as scatter with matplotlib library.
    """

    def __init__(self, fig_size=(13, 8), dpi=200, face_color='w', edge_color='k', title_fontsize=16):
        """
        Initialize the figure template with Nanoz style.

        Parameters
        ----------
        fig_size : tuple of float, default=(13, 8)
            Width and height of the figure in inches.
        dpi : float, default=200
            The resolution of the figure in dots-per-inch.
        face_color : matplotlib color, default='w'
            The background color of the figure.
            For all color, see https://matplotlib.org/stable/gallery/color/named_colors.html.
        edge_color : matplotlib color, default='k'
            The border color of the figure.
            For all color, see https://matplotlib.org/stable/gallery/color/named_colors.html.
        title_fontsize : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, default=16
            Font size of the figure title.
            If a float, the font size in points. The string values denote sizes relative to the default font size.
        """
        self.fig_size = fig_size
        self.dpi = dpi
        self.face_color = face_color
        self.edge_color = edge_color
        self.title_fontsize = title_fontsize
        self.colors_line = ['b', 'r', 'g', 'y', 'm', 'c']
        self.line_styles = ['-', ':', '--']
        self.marker_styles = ['o', '^', '*']

        self._fig = None
        self._ax = None
        self._ax_right = None
        self._ax_ins = None
        self.reset_fig()

    @property
    def formats_line(self):
        """
        Get a list of format string based on colors_line and line_styles, e.g. '-r' for red solid line.

        Returns
        -------
        list of str
            List of format string.
        """
        return [''.join(x) for x in list(itertools.product(self.line_styles, self.colors_line))]

    @property
    def formats_marker(self):
        """
        Get a list of format string based on colors_line and marker_styles, e.g. 'ro' for red circles.

        Returns
        -------
        list of str
            List of format string.
        """
        return [''.join(x) for x in list(itertools.product(self.marker_styles, self.colors_line))]

    def reset_fig(self):
        """
        Reset the figure.
        """
        plt.close(self._fig)
        self._fig, self._ax = plt.subplots(
            figsize=self.fig_size,
            dpi=self.dpi,
            facecolor=self.face_color,
            edgecolor=self.edge_color
        )
        self._ax_right = None
        if is_notebook():
            self._fig.canvas.toolbar_visible = 'fade-in-fade-out'

    def _save_fig(self, save_path, svg=True):
        """
        Save the figure as an image files (png format and svg in option).

        Parameters
        ----------
        save_path : Path
            Image saving folder path.
        svg: bool, default=True
            If is True, save the figure as a png format also.
        """
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            logging.debug('Figure saved at {0}'.format(save_path))
            if svg:
                svg_path = Path(save_path.parent, save_path.stem + '.svg')
                plt.savefig(svg_path, format='svg')
                logging.debug('Figure saved at {0}'.format(svg_path))

    def plot_one_y_axis(self, y, x=None, title='', y_label='', x_label='', x_scale='linear',
                        x_ticks=None, x_rotate=0, y_lim=None, inset=None, show_grid=True, save_path=None):
        """
        Plot y versus x as lines with matplotlib library.

        Parameters
        ----------
        y : dict
            Dictionary with name of the y as 'keys' and values of y as 'values'.
        x : list, default=None
            Values of x.
        title : str, default=''
            Title of the figure.
        y_label : str, default=''
            Title of the y-axis.
        x_label : str, default=''
            Title of the x-axis.
        x_scale : {'linear', 'log', 'symlog','logit'}, default='linear'
            The axis scale type to apply on the x-axis.
        x_ticks : array-like, default=None
            List of tick locations and labels of the x-axis.
        x_rotate : float or {'vertical', 'horizontal'}, default=0
            Rotation in degrees to apply at the x-axis ticks.
        y_lim : int or list of int, default=None
            Set the y-limits of the left axis.
            If is an integer, set the bottom value. If is a list set the bottom and top values.
        inset : list, default=None
            Add zoom plot on the current plot.
        show_grid : bool, default=True
            If is True, show the grid lines.
        save_path : Path, default=None
            Image saving folder path. If None, doesn't save the figure.
        """
        self._fig.suptitle(title, fontsize=self.title_fontsize)
        if x is None:
            for plot_name, fmt in zip(list(y), self.formats_line):
                self._ax.plot(y[plot_name], fmt, label=plot_name)
        else:
            for plot_name, fmt in zip(list(y), self.formats_line):
                self._ax.plot(x, y[plot_name], fmt, label=plot_name)
        self._ax.set(xlabel=x_label, ylabel=y_label)
        self._ax.legend(bbox_to_anchor=(0.6, 1.07), ncol=3, fancybox=True, shadow=True)
        _set_y_lim(self._ax, y_lim)
        _set_x_scale(self._ax, x_scale)
        _set_x_ticks(self._ax, x_ticks, x_rotate)
        _set_grid(self._ax, show_grid)
        self._save_fig(save_path)
        if inset is not None:
            self._ax_ins = self._ax.inset_axes([0.5, 0.25, 0.47, 0.47])
            if x is None:
                for plot_name, fmt in zip(list(y), self.formats_line):
                    self._ax.plot(y[plot_name], fmt, label=plot_name)
            else:
                for plot_name, fmt in zip(list(y), self.formats_line):
                    self._ax.plot(x, y[plot_name], fmt, label=plot_name)
            self._ax_ins.set_xlim(inset[0], inset[1])
            self._ax_ins.set_ylim(inset[2], inset[3])
            _set_x_scale(self._ax_ins, x_scale)
            _set_grid(self._ax_ins, show_grid)
            self._ax.indicate_inset_zoom(self._ax_ins, edgecolor="grey")
            self._save_fig(Path(save_path.parent, save_path.stem+'_insaxis'+''.join(save_path.suffixes)))

    def plot_double_y_axis(self, *y, x=None, title='', y_label=('', ''), x_label='', x_scale='linear',
                           x_ticks=None, x_rotate=0, y_lim=None, y_lim_right=None, show_grid=True, save_path=None):
        """
        Comparison of plots y versus x as lines on two y-axis with matplotlib library.

        Parameters
        ----------
        y : dict
            Dictionary with name of the y as 'keys' and values of y as 'values'.
        x : list, default=None
            Values of x.
        title : str, default=''
            Title of the figure.
        y_label : tuple of str, default=('', '')
            Title of the left and right y-axis.
        x_label : str, default=''
            Title of the x-axis.
        x_scale : {'linear', 'log', 'symlog','logit'}, default='linear'
            The axis scale type to apply on the x-axis.
        x_ticks : array-like, default=None
            List of tick locations and labels of the x-axis.
        x_rotate : float or {'vertical', 'horizontal'}, default=0
            Rotation in degrees to apply at the x-axis ticks.
        y_lim : int or list of int, default=None
            Set the y-limits of the left axis.
            If is an integer, set the bottom value. If is a list set the bottom and top values.
        y_lim_right : int or list of int, default=None
            Set the y-limits of the right axis.
            If is an integer, set the bottom value. If is a list set the bottom and top values.
        show_grid : bool, default=True
            If is True, show the grid lines.
        save_path : Path, default=None
            Image saving folder path. If None, doesn't save the figure.
        """
        self._fig.suptitle(title, fontsize=self.title_fontsize)
        self._ax_right = self._ax.twinx()
        if x is None:
            for plot_name, fmt in zip(list(y[0]), self.formats_line):
                self._ax.plot(y[0][plot_name], fmt, label=plot_name)
            for plot_name, fmt in zip(list(y[1]), reversed(self.formats_line)):
                self._ax_right.plot(y[1][plot_name], fmt, label=plot_name)
        else:
            for plot_name, fmt in zip(list(y[0]), self.formats_line):
                self._ax.plot(x, y[0][plot_name], fmt, label=plot_name)
            for plot_name, fmt in zip(list(y[1]), reversed(self.formats_line)):
                self._ax_right.plot(x, y[1][plot_name], fmt, label=plot_name)
        self._ax.set(xlabel=x_label, ylabel=y_label[0])
        self._ax_right.set(ylabel=y_label[1])
        self._ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.07), ncol=3, fancybox=True, shadow=True)
        self._ax_right.legend(loc='upper right', bbox_to_anchor=(0.9, 1.07), ncol=3, fancybox=True, shadow=True)
        self._ax.set_ylim(y_lim)
        self._ax_right.set_ylim(y_lim_right)
        _set_x_scale(self._ax, x_scale)
        _set_x_ticks(self._ax, x_ticks, x_rotate)
        _set_grid(self._ax, show_grid)
        self._save_fig(save_path)

    def scatter_one_y_axis(self, y, x=None, title='', y_label='', x_label='', x_scale='linear',
                           x_ticks=None, x_rotate=0, y_lim=None, show_grid=True, save_path=None):
        """
        Plot y versus x as scatter with matplotlib library.

        Parameters
        ----------
        y : dict
            Dictionary with name of the y as 'keys' and values of y as 'values'.
        x : list, default=None
            Values of x.
        title : str, default=''
            Title of the figure.
        y_label : str, default=''
            Title of the y-axis.
        x_label : str, default=''
            Title of the x-axis.
        x_scale : {'linear', 'log', 'symlog','logit'}, default='linear'
            The axis scale type to apply on the x-axis.
        x_ticks : array-like, default=None
            List of tick locations and labels of the x-axis.
        x_rotate : float or {'vertical', 'horizontal'}, default=0
            Rotation in degrees to apply at the x-axis ticks.
        y_lim : int or list of int, default=None
            Set the y-limits of the left axis.
            If is an integer, set the bottom value. If is a list set the bottom and top values.
        show_grid : bool, default=True
            If is True, show the grid lines.
        save_path : Path, default=None
            Image saving folder path. If None, doesn't save the figure.
        """
        self._fig.suptitle(title, fontsize=self.title_fontsize)
        if x is None:
            for plot_name, fmt in zip(list(y), self.formats_marker):
                self._ax.plot(y[plot_name], fmt, label=plot_name, s=1)
        else:
            for plot_name, fmt in zip(list(y), self.formats_marker):
                self._ax.plot(x, y[plot_name], fmt, label=plot_name)
        self._ax.set(xlabel=x_label, ylabel=y_label)
        self._ax.legend(bbox_to_anchor=(0.6, 1.07), ncol=3, fancybox=True, shadow=True)
        _set_y_lim(self._ax, y_lim)
        _set_x_scale(self._ax, x_scale)
        _set_x_ticks(self._ax, x_ticks, x_rotate)
        _set_grid(self._ax, show_grid)
        self._save_fig(save_path, svg=False)

    def plot_pandas_density(self, df, title='', show_grid=True, save_path=None):
        """
        Plot frequency as histogram and the Kernel Density Estimation with pandas library.

        Parameters
        ----------
        df : Series or DataFrame
            The object for which the method is called.
        title : str, default=''
            Title of the figure.
        show_grid : bool, default=True
            If is True, show the grid lines.
        save_path : Path, default=None
            Image saving folder path. If None, doesn't save the figure.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them
            The Matplotlib axis of the data histogram calculated by pandas.
        """
        self._fig.suptitle(title, fontsize=self.title_fontsize)
        if isinstance(df.index, RangeIndex):
            axis = df.plot(kind='hist', ax=self._ax, style=self.colors_line[0])
        elif isinstance(df.index, IntervalIndex):
            axis = df.plot(kind='bar', rot=0, ax=self._ax, style=self.colors_line[0])
            axis.set_xticklabels([f'{i.left}\n to \n{i.right}' for i in df.index])
        else:
            axis = None
        self._ax.set(ylabel='Frequency')
        try:
            self._ax_right = self._ax.twinx()
            df.plot(kind='kde', ax=self._ax_right, style=self.colors_line[1])
            self._ax_right.set(ylabel='Density')
            self._ax_right.legend(loc='upper right', bbox_to_anchor=(0.9, 1.07), ncol=3, fancybox=True, shadow=True)
        except LinAlgError as lae:
            logging.debug(f'{lae} for {title} plot.')
        _set_grid(self._ax, show_grid)
        self._ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.07), ncol=3, fancybox=True, shadow=True)
        self._save_fig(save_path)
        return axis

    def comparison_pandas_density(self, dict_df, title='', show_grid=True, save_path=None):
        """
        Plot the Kernel Density Estimation with pandas library.

        Parameters
        ----------
        dict_df : dict
            Dictionary of pandas Series or DataFrame.
        title : str, default=''
            Title of the figure.
        show_grid : bool, default=True
            If is True, show the grid lines.
        save_path : Path, default=None
            Image saving folder path. If None, doesn't save the figure.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them
            The Matplotlib axis of the data histogram calculated by pandas.
        """
        self._fig.suptitle(title, fontsize=self.title_fontsize)
        self._ax.set(ylabel='Density')
        try:
            for df, color in zip(dict_df.keys(), self.colors_line):
                dict_df[df].plot(kind='kde', ax=self._ax, style=color, label=df)
        except LinAlgError as lae:
            logging.debug(f'{lae} for {title} plot.')
        _set_grid(self._ax, show_grid)
        self._ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.07), ncol=3, fancybox=True, shadow=True)
        self._save_fig(save_path)
        return self._ax


def save_data_distribution(datasets, save_paths):
    colors = px.colors.qualitative.Plotly
    columns = datasets["test"].data.columns
    fig = make_subplots(rows=len(columns), cols=2, vertical_spacing=0.01,
                        subplot_titles=[val for val in columns for _ in (0, 1)])

    for i, (name, dataset) in enumerate(datasets.items()):
        color = colors[i]
        for j, column in enumerate(columns):
            legend = True if j == 0 else False
            data = dataset.data[column]
            hist, edges = np.histogram(data[~np.isnan(data)], bins=20)
            edges_width = np.mean(np.diff(edges))
            center_edges = [left+(right-left)/2 for left, right in zip(edges[::], edges[1::])]
            fig.add_trace(
                go.Bar(x=center_edges,
                       y=hist,
                       name=name,
                       legendgroup=name,
                       showlegend=False,
                       width=edges_width/5,
                       marker=dict(color=color),
                       opacity=0.5
                       ),
                row=j+1, col=1
            )
            try:
                x_pdf, y_pdf = gaussian_kde(data)
                fig.add_trace(
                    go.Scatter(
                        x=x_pdf,
                        y=y_pdf,
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=legend,
                        marker=dict(color=color)
                    ),
                    row=j+1, col=2
                )
            except:
                logging.debug(f"Exception when computing the gaussian_kde of the {name} dataset "
                              f"for the {column} column.")

    update_menu = go.layout.Updatemenu(
        type="buttons",
        showactive=True,
        buttons=[{
            "method": "relayout",
            "args": [{"template": pio.templates["plotly_dark"]}],
            "args2": [{"template": pio.templates["plotly_white"]}],
            "label": "White / Dark"
        }],
        bgcolor="#BEBEBE",
        font={"color": "#000000"},
        x=1, y=1.005,
        xanchor="center"
    )

    # Update size of subplot titles
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=20)

    fig.update_layout(
        height=600*len(columns),
        title_text=f"Distribution of datasets",
        title_font=dict(size=30),
        template="plotly_dark",
        updatemenus=[update_menu]
    )
    fig.write_html(save_paths)


def white_dark_button(x=1, y=1.1):
    update_menu = go.layout.Updatemenu(
        type="buttons",
        showactive=True,
        buttons=[{
            "method": "update",
            "args": [{"header.fill": [{"color": "#121212"}], "header.font": [{"color": "#FFFFFF"}],
                      "cells.fill": [{"color": "#121212"}], "cells.font": [{"color": "#FFFFFF"}]},
                     {"template": pio.templates["plotly_dark"]}],
            "args2": [{"header.fill": [{"color": "#FFFFFF"}], "header.font": [{"color": "#121212"}],
                       "cells.fill": [{"color": "#FFFFFF"}], "cells.font": [{"color": "#121212"}]},
                      {"template": pio.templates["plotly_white"]}],
            "label": "White / Dark"
        }],
        bgcolor="#BEBEBE",
        font={"color": "#000000"},
        x=x, y=y,
        xanchor="center"
    )
    return update_menu


class DashboardFactory:
    @staticmethod
    def create_dashboard(algo, **kwargs):
        algo_type = AvailableAlgorithm.get_type(algo)
        if algo_type == "regression":
            logging.debug(f"Creating RegressionDashboard with {kwargs}")
            return RegressionDashboard(**kwargs)
        elif algo_type == "classification":
            logging.debug(f"Creating ClassificationDashboard with {kwargs}")
            return ClassificationDashboard(**kwargs)
        else:
            raise ValueError(f"Invalid algorithm: {algo}")


class Dashboard:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.evaluator = kwargs.get("evaluator")

        self.figs = {}
        self.color_lines = px.colors.qualitative.Plotly
        self.color_lines.extend(px.colors.qualitative.G10)
        self.font_size = 18
        self.plot_height = 800

    def save_dashboards(self, save_path, name):
        for dashboard_name, dashboard in self.figs.items():
            dashboard.write_html(Path(save_path, name + "_" + dashboard_name).with_suffix(".html"))


class RegressionDashboard(Dashboard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = kwargs.get("sample_size", 100000)

        self.figs["global"] = self._create_global_dashboard()
        if self.evaluator.intervals:
            for id_target, target_name in enumerate(self.evaluator.targets_name):
                self.figs[target_name+"_per_interval"] = self._create_per_interval_dashboard(target_name, id_target)

    def _create_global_dashboard(self):
        subplot_titles = ["Model performance metrics", "Predicted vs Ground Truth",
                          "Residuals distribution", "Percent error density"]
        specs = [[{"type": "table"}, {"type": "scatter"}],
                 [{"type": "xy", "colspan": 2}, None],
                 [{"type": "xy", "colspan": 2}, None]]

        fig = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.075, specs=specs)

        # Model performance metrics
        headers = self.evaluator.targets_name.copy()
        headers.insert(0, "metrics")
        cells = self.evaluator.target_performances.astype(float).round(3).values.tolist().copy()
        cells.insert(0, self.evaluator.target_performances.columns)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                ),
                cells=dict(
                    values=cells,
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                )
            ),
            row=1, col=1
        )

        # Predicted vs Ground Truth random dash line
        gt_min = self.evaluator.ground_truth.min()
        gt_max = self.evaluator.ground_truth.max()
        pred_min = self.evaluator.prediction.min()
        pred_max = self.evaluator.prediction.max()
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=gt_min, x1=gt_max, y0=gt_min, y1=gt_max,
            row=1, col=2
        )

        # Random sampling
        sample_size, target_number = self.evaluator.prediction.shape
        targets_idx = {}
        for target in self.evaluator.targets_name:
            targets_idx[target] = np.arange(sample_size)
        random_index = targets_idx.copy()
        all_samples = sample_size * target_number
        if all_samples > self.sample_size:
            plotted_samples = 0
            subsample_size = max(1, self.sample_size // target_number)
            for target in self.evaluator.targets_name:
                random_index[target] = np.random.choice(sample_size, size=subsample_size, replace=False)
                plotted_samples += random_index[target].shape[0]
        else:
            plotted_samples = sample_size * target_number

        # Loop on targets
        for i, target in enumerate(self.evaluator.targets_name):
            gt = self.evaluator.ground_truth[[random_index[target]], i].squeeze(axis=0)
            pred = self.evaluator.prediction[[random_index[target]], i].squeeze(axis=0)
            residuals = gt - pred
            percent_error = residuals / gt * 100
            percent_error[percent_error > 100] = 99.99
            percent_error[percent_error < -100] = -99.99

            # Predicted vs Ground Truth
            fig.add_trace(
                go.Scatter(
                    x=gt,
                    y=pred,
                    mode="markers",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{target}",
                    legendgroup=target,
                    showlegend=True
                ),
                row=1, col=2
            )

            # Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    marker=dict(color=self.color_lines[i]),
                    name=f"{target}",
                    legendgroup=target,
                    showlegend=False
                ),
                row=2, col=1
            )

            # Percent error density
            try:
                x_pdf, y_pdf = gaussian_kde(percent_error)
                fig.add_trace(
                    go.Scatter(
                        x=x_pdf,
                        y=y_pdf,
                        mode="lines",
                        marker=dict(color=self.color_lines[i]),
                        name=f"{target}",
                        legendgroup=target,
                        showlegend=False
                    ),
                    row=3, col=1
                )
            except:
                fig.add_trace(
                    go.Histogram(
                        x=percent_error,
                        histnorm="density",
                        marker=dict(color=self.color_lines[i]),
                        name=f"{target}",
                        legendgroup=target,
                        showlegend=False
                    ),
                    row=3, col=1
                )

        # Updates axis titles
        fig.update_xaxes(title_text="Ground truth", range=[gt_min, gt_max], row=1, col=2)
        fig.update_yaxes(title_text="Predictions", range=[pred_min, pred_max], row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", type="log", row=2, col=1)
        fig.update_xaxes(title_text="Percent error (%)", range=[-100, 100], row=3, col=1)
        fig.update_yaxes(title_text="Density", row=3, col=1)

        # Update size of subplot titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=self.font_size + 2)

        # Update layout
        fig.update_layout(
            height=3 * self.plot_height,
            margin=dict(t=400),
            title=go.layout.Title(
                text=f"<B>Regression dashboard - {self.name} dataset</B><br>"
                     f"<sup></sup><br>"
                     f"<sup>Targets: {' - '.join(self.evaluator.targets_name)}</sup><br>"
                     f"<sup>Random sampling: {plotted_samples} / {all_samples} "
                     f"({100*plotted_samples/all_samples:.3f} %)</sup>",
                xref="paper",
                x=0,
                font=dict(size=30)
            ),
            font=dict(size=self.font_size),
            legend=dict(y=0.55),
            template="plotly_dark",
            updatemenus=[white_dark_button()]
        )

        return fig

    def _create_per_interval_dashboard(self, target_name, id_target):
        ground_truth = self.evaluator.ground_truth[:, id_target]
        prediction = self.evaluator.prediction[:, id_target]
        metrics_of_interval = self.evaluator.intervals_performances[target_name].columns.tolist()
        intervals = [str(interval) for interval in self.evaluator.intervals]

        # Figure
        subplot_titles = ["Model performance metrics per interval", "Predicted vs Ground Truth per interval",
                          "Residuals distribution per interval", "Percent error density per interval"]
        specs = [[{"type": "table"}, {"type": "scatter"}],
                 [{"type": "xy", "colspan": 2}, None],
                 [{"type": "xy", "colspan": 2}, None]]
        fig = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.075, specs=specs)

        # Random sampling
        intervals_idx = {}
        interval_samples = []
        for interval in self.evaluator.intervals:
            intervals_idx[str(interval)] = np.where((ground_truth > interval[0]) & (ground_truth <= interval[1]))[0]
            interval_samples.append(intervals_idx[str(interval)].shape[0])
        all_samples = sum(interval_samples)
        random_idx = intervals_idx.copy()
        if all_samples > self.sample_size:
            max_subsample_size = max(1, self.sample_size // len(intervals))
            copy_interval_samples = []
            for interval, interval_size in zip(intervals, interval_samples):
                if interval_size <= 0:
                    copy_interval_samples.append(0)
                    continue
                subsample_size = min(max_subsample_size, interval_size)
                random_idx[interval] = np.random.choice(intervals_idx[interval], size=subsample_size, replace=False)
                copy_interval_samples.append(random_idx[interval].shape[0])
            interval_samples = copy_interval_samples
        plotted_samples = sum(interval_samples)

        # Model performance metrics per interval
        df_intervals_performances = self.evaluator.intervals_performances[target_name].astype(float).round(3).copy()
        df_intervals_performances.insert(0, "Sampling", interval_samples)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Intervals"] + ["Sampling"] + metrics_of_interval,
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                ),
                cells=dict(
                    values=df_intervals_performances.reset_index().transpose().values.tolist(),
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                )
            ),
            row=1, col=1
        )

        # Predicted vs Ground Truth dash line
        gt_min = ground_truth.min()
        gt_max = ground_truth.max()
        pred_min = prediction.min()
        pred_max = prediction.max()
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=gt_min, x1=gt_max, y0=gt_min, y1=gt_max,
            row=1, col=2
        )

        # Loop on intervals
        for i, interval in enumerate(intervals):
            if intervals_idx[interval].shape[0] <= 0:
                continue
            gt = ground_truth[random_idx[interval]]
            pred = prediction[random_idx[interval]]
            residuals = gt - pred
            percent_error = residuals / gt * 100
            percent_error[percent_error > 100] = 99.99
            percent_error[percent_error < -100] = -99.99

            # Predicted vs Ground Truth
            fig.add_trace(
                go.Scatter(
                    x=gt,
                    y=pred,
                    mode="markers",
                    marker=dict(color=self.color_lines[i]),
                    name=interval,
                    legendgroup=i,
                    showlegend=True
                ),
                row=1, col=2
            )

            # Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    marker=dict(color=self.color_lines[i]),
                    name=interval,
                    legendgroup=i,
                    showlegend=False
                ),
                row=2, col=1
            )

            # Percent error density
            try:
                x_pdf, y_pdf = gaussian_kde(percent_error)
                fig.add_trace(
                    go.Scatter(
                        x=x_pdf,
                        y=y_pdf,
                        mode="lines",
                        marker=dict(color=self.color_lines[i]),
                        name=interval,
                        legendgroup=i,
                        showlegend=False
                    ),
                    row=3, col=1
                )
            except:
                fig.add_trace(
                    go.Histogram(
                        x=percent_error,
                        histnorm="density",
                        marker=dict(color=self.color_lines[i]),
                        name=interval,
                        legendgroup=i,
                        showlegend=False
                    ),
                    row=3, col=1
                )

        # Updates axis titles
        fig.update_xaxes(title_text="Ground truth", range=[gt_min, gt_max], row=1, col=2)
        fig.update_yaxes(title_text="Predictions", range=[pred_min, pred_max], row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", type="log", row=2, col=1)
        fig.update_xaxes(title_text="Percent error (%)", range=[-100, 100], row=3, col=1)
        fig.update_yaxes(title_text="Density", row=3, col=1)

        # Update size of subplot titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=self.font_size + 2)

        # Update layout
        fig.update_layout(
            height=3 * self.plot_height,
            margin=dict(t=400),
            title=go.layout.Title(
                text=f"<B>Regression dashboard - {self.name} dataset</B><br>"
                     f"<sup></sup><br>"
                     f"<sup>Target: {target_name}</sup><br>"
                     f"<sup>Random sampling: {plotted_samples} / {all_samples} "
                     f"({100 * plotted_samples / all_samples:.3f} %)</sup>",
                xref="paper",
                x=0,
                font=dict(size=30)
            ),
            font=dict(size=self.font_size),
            legend=dict(y=0.55),
            template="plotly_dark",
            updatemenus=[white_dark_button()]
        )
        return fig


class ClassificationDashboard(Dashboard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hover_template = {
            "ROC_th": "<b>TPR</b>: %{y:.3f}</b><br><b>FPR: %{x:.3f}</b><br><i>threshold: %{customdata:.3f}</i>",
            "ROC": "<b>TPR</b>: %{y:.3f}</b><b>FPR: %{x:.3f}</b><br>",
            "PR_th": "<b>Precision</b>: %{y:.3f}</b><br><b>Recall: %{x:.3f}</b><br><i>threshold: %{customdata:.3f}</i>",
            "PR": "<b>Precision</b>: %{y:.3f}</b><b>Recall: %{x:.3f}</b><br>",
        }

        self.figs["global"] = self._create_global_dashboard()
        self.figs["per_class"] = self._create_per_class_dashboard()

    @staticmethod
    def _cm_annotation(confusion_matrix, normalized_cm, classes):
        th_color = confusion_matrix.min() + (confusion_matrix.max()-confusion_matrix.min())/2
        class_names = [f"({c[0]}-{c[1]}]" for c in classes]
        font_size = max(20-len(class_names), 12)
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    dict(
                        x=class_names[j],
                        y=class_names[i],
                        text=str(confusion_matrix[i, j]) + "<br>" + f"{normalized_cm[i, j]:.2%}",
                        showarrow=False,
                        font=dict(color="black" if confusion_matrix[i, j] > th_color else "white", size=font_size)
                    )
                )
        return annotations, class_names

    def _create_global_dashboard(self):
        subplot_titles = ["Model performance metrics", "Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
        specs = [[{"type": "table"}, {"type": "heatmap"}], [{"type": "scatter"}, {"type": "scatter"}]]

        fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.1, specs=specs)

        # Model performance metrics
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["", ""],
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                ),
                cells=dict(
                    values=[self.evaluator.performances.columns,
                            self.evaluator.performances.astype(float).round(3).values[0]],
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                )
            ),
            row=1, col=1
        )

        # Confusion Matrix
        cm_annotations, class_names = self._cm_annotation(self.evaluator.confusion_matrix,
                                                          self.evaluator.normalized_cm,
                                                          self.evaluator.intervals)
        fig.add_trace(
            go.Heatmap(
                z=self.evaluator.confusion_matrix,
                x=class_names,
                y=class_names,
                colorscale=[[0.0, "#121212"], [0.09090909090909091, "#191933"],
                            [0.18181818181818182, "#2C2A57"], [0.2727272727272727, "#3A3C7D"],
                            [0.36363636363636365, "#3E53A0"], [0.45454545454545453, "#3E6DB2"],
                            [0.5454545454545454, "#4886BB"], [0.6363636363636364, "#599FC4"],
                            [0.7272727272727273, "#72B8CD"], [0.8181818181818182, "#95CFD8"],
                            [0.9090909090909091, "#C0E5E8"], [1.0, "#FFFFFF"]],  # custom_ice
                showscale=True,
                name=f"",
                showlegend=False,
                colorbar=dict(len=0.495, y=1, yanchor="top"),
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Prediction", row=1, col=2)
        fig.update_yaxes(title_text="Ground truth", row=1, col=2)

        # ROC Curve and Precision-Recall Curve
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=0, x1=1, y0=0, y1=1,
            row=2, col=1
        )
        for i, name in enumerate(["micro", "macro"]):
            fig.add_trace(
                go.Scatter(
                    x=self.evaluator.fpr[name],
                    y=self.evaluator.tpr[name],
                    customdata=self.evaluator.roc_th[name] if name == "micro" else [],
                    mode="lines",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{name}",
                    legendgroup=name,
                    showlegend=True,
                    hovertemplate=self.hover_template["ROC_th"] if name == "micro" else self.hover_template["ROC"]
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.evaluator.recall[name],
                    y=self.evaluator.precision[name],
                    customdata=self.evaluator.pr_th[name] if name == "micro" else [],
                    mode="lines",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{name}",
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=self.hover_template["PR_th"] if name == "micro" else self.hover_template["PR"]
                ),
                row=2, col=2
            )
        fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=2, col=1)
        fig.update_xaxes(title_text="Recall", range=[0, 1], row=2, col=2)
        fig.update_yaxes(title_text="Precision", range=[0, 1], row=2, col=2)

        # Update size of subplot titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=self.font_size + 2)

        # Update layout
        fig.update_layout(
            height=2 * self.plot_height,
            margin=dict(t=400),
            title=go.layout.Title(
                text=f"<B>Classification dashboard - {self.name} dataset</B><br>"
                     f"<sup></sup><br>"
                     f"<sup>Targets: {' - '.join(self.evaluator.targets_name)}</sup><br>"
                     f"<sup>Ground truth size: {self.evaluator.ground_truth.shape}",
                xref="paper",
                x=0,
                font=dict(size=30)
            ),
            font=dict(size=self.font_size),
            legend=dict(y=0.4),
            template="plotly_dark",
            updatemenus=[white_dark_button()]
        )

        # Add annotations for confusion matrix at the end of the figure
        for annotation in cm_annotations:
            fig.add_annotation(annotation)

        return fig

    def _create_per_class_dashboard(self):
        metrics_of_class = self.evaluator.intervals_performances.columns.tolist()
        classes = self.evaluator.intervals_performances.index
        n_classes = len(classes)

        subplot_titles = ["Model performance metrics per class", "Performances per class",
                          "ROC Curve per class", "Precision-Recall Curve per class"]
        specs = [[{"type": "table"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]]

        fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.1, specs=specs)

        # Model performance metrics per class
        df_intervals_performances = self.evaluator.intervals_performances.astype(float).round(3)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Classes"] + metrics_of_class,
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                ),
                cells=dict(
                    values=df_intervals_performances.reset_index().transpose().values.tolist(),
                    font={"size": self.font_size, "color": "white"},
                    fill={"color": "#121212"},
                    height=50,
                    align="left"
                )
            ),
            row=1, col=1
        )

        # Metrics per classes
        for i, metric in enumerate(metrics_of_class):
            fig.add_trace(
                go.Scatter(
                    x=classes,
                    y=self.evaluator.intervals_performances[metric],
                    mode="lines",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{metric}",
                    showlegend=True
                ),
                row=1, col=2
            )
        fig.update_xaxes(title_text="Classes", row=1, col=2)
        fig.update_yaxes(title_text="Performances", range=[0, 1], row=1, col=2)

        # ROC Curve and Precision-Recall Curve per classes
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=0, x1=1, y0=0, y1=1,
            row=2, col=1
        )
        for i in range(n_classes):
            fig.add_trace(
                go.Scatter(
                    x=self.evaluator.fpr[i],
                    y=self.evaluator.tpr[i],
                    customdata=self.evaluator.roc_th[i],
                    mode="lines",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{classes[i]}",
                    legendgroup=i,
                    showlegend=True,
                    hovertemplate=self.hover_template["ROC_th"]
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.evaluator.recall[i],
                    y=self.evaluator.precision[i],
                    customdata=self.evaluator.pr_th[i],
                    mode="lines",
                    marker=dict(color=self.color_lines[i]),
                    name=f"{classes[i]}",
                    legendgroup=i,
                    showlegend=False,
                    hovertemplate=self.hover_template["PR_th"]
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=2, col=1)
            fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=2, col=1)
            fig.update_xaxes(title_text="Recall", range=[0, 1], row=2, col=2)
            fig.update_yaxes(title_text="Precision", range=[0, 1], row=2, col=2)

        # Update size of subplot titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=self.font_size + 2)

        # Update layout
        fig.update_layout(
            height=2 * self.plot_height,
            margin=dict(t=400),
            title=go.layout.Title(
                text=f"<B>Classification dashboard - {self.name} dataset</B><br>"
                     f"<sup></sup><br>"
                     f"<sup>Target: {' - '.join(self.evaluator.targets_name)}</sup><br>"
                     f"<sup>Ground truth size: {self.evaluator.ground_truth.shape[0]}",
                xref="paper",
                x=0,
                font=dict(size=30)
            ),
            font=dict(size=self.font_size),
            legend=dict(y=0.75),
            template="plotly_dark",
            updatemenus=[white_dark_button()]
        )

        return fig
