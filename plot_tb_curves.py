# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import collections
import os
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import matplotlib
import matplotlib.pyplot as plt
# Style gallery: https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
plt.style.use('seaborn-dark')


def namedtuple(typename, field_names, default_value=None, default_values=()):
    """namedtuple with default value.
    Args:
        typename (str): type name of this namedtuple.
        field_names (list[str]): name of each field.
        default_value (Any): the default value for all fields.
        default_values (list|dict): default value for each field.
    Returns:
        the type for the namedtuple
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (default_value, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def _compute_y_interval(interval_mode, ys):
    """Given several aligned y curves, compute a y value interval at each x.
    ``interval_mode`` should be one of the four options: 1) "std", 2) "minmax",
    and 3) "CI_X". The last one means confidence interval, where 'X'
    should be one of ``(80,85,90,95,99)`` indicating the confidence level (percentage).
    Returns:
        tuple - a triplet of mean of y, lower interval bound, and upper interval bound
    """
    CI_Z = {"80": 1.282, "85": 1.440, "90": 1.645, "95": 1.960, "99": 2.576}

    def _get_ci_z(interval_mode):
        """Get the corresponding Z value used in confidence interval computation."""
        # mode -> "CI_X" where X is the percentage
        ci_level = interval_mode.split("_")[1]
        assert ci_level in CI_Z, "Invalid level value %s!" % ci_level
        return CI_Z[ci_level]

    y = np.array(list(map(np.mean, zip(*ys))))
    std = np.array(list(map(np.std, zip(*ys))))
    if interval_mode == "std":
        min_y, max_y = y - std, y + std
    elif interval_mode == "minmax":
        min_y = np.array(list(map(np.min, zip(*ys))))
        max_y = np.array(list(map(np.max, zip(*ys))))
    elif interval_mode.startswith("CI_"):
        z = _get_ci_z(interval_mode)
        margin_err = std / np.sqrt(len(ys)) * z
        min_y, max_y = y - margin_err, y + margin_err
    else:
        raise ValueError("Invalid interval mode: %s" % interval_mode)

    return y, min_y, max_y


class MeanCurve(
        namedtuple(
            "MeanCurve",
            ['x', 'y', 'min_y', 'max_y', 'ay', 'min_ay', 'max_ay', 'name'],
            default_value=None)):
    @classmethod
    def from_curves(cls, x, ys, interval_mode="std", name="MeanCurve"):
        """Compute various curve statistics from a set of individual curves ``ys``
        and a common ``x``, and create a class instance.
        Args:
            x (np.array): x steps
            ys (list[np.array]): a list of curves
            interval_mode (str): mode for computing error margin around the mean
                y curve. Should be one of the four options: 1) "std", 2) "minmax",
                and 3) "CI_X". The last one means confidence interval, where 'X'
                should be one of ``(80,85,90,95,99)`` indicating the confidence
                level (percentage).
            name (str):
        """
        # mean curve, lower and upper curve
        y, min_y, max_y = _compute_y_interval(interval_mode, ys)
        ays = [np.mean(y, keepdims=True) for y in ys]
        # mean average_y, lower and upper average_y
        # average_y can be used to indicate the changing trend of y
        ay, min_ay, max_ay = map(lambda z: z.squeeze(-1),
                                 _compute_y_interval(interval_mode, ays))
        return cls(
            x=x,
            y=y,
            min_y=min_y,
            max_y=max_y,
            ay=ay,
            min_ay=min_ay,
            max_ay=max_ay,
            name=name)

    def final_y(self, N=1):
        return tuple(
            map(lambda y: np.mean(y[-N:]), (self.y, self.min_y, self.max_y)))


class MeanCurveReader(object):
    """Read and compute a ``MeanCurve`` from one or multiple TB event files. A
    ``MeanCurveReader`` is suitable for one method on one task with multiple runs.
    """

    def _get_metric_name(self):
        raise NotImplementedError()

    @property
    def x_label(self):
        raise NotImplementedError()

    @property
    def y_label(self):
        raise NotImplementedError()

    def __init__(self,
                 event_file,
                 x_steps=None,
                 x_scale='step',
                 name="MeanCurveReader",
                 smoothing=None,
                 interval_mode="std"):
        """
        Args:
            event_file (str|list[str]): a string or a list of strings where
                each should point to a valid TB dir, e.g., ending with
                "eval/" or "train/". The curves of these files will be averaged.
                It's the user's responsibility to ensure that it's meaningful to
                group these event files and show their mean and variance.
            x_steps (list[int]): we support merging curves that have different
                :math:`x` into a ``MeanCurve``. For example, if there are three
                curves:
                .. code-block:: python
                    curve1 x: (1, 9),
                    curve2 x: (0, 10),
                    curve3 x: (0, 8),
                then the merged ``MeanCurve`` will have :math:`(1, 8)` as the
                final :math:`x` range. Each curve's new :math:`y` values will
                be interpolated w.r.t. this common :math:`x` range approperiately
                given their original :math:`y=f(x)` curve. The common :math:`x`
                range will be automatically determined as in the example if this
                argument ``x_steps==None``. Alternatively, the user can specify
                a pre-defined list of integers for interpolation.
            x_scale (str): ``step`` or ``relative``.
            name (str): name of the mean curve.
            smoothing (int | float): if None, no smoothing is applied; if int,
                it's the window width of a Savitzky-Golay filter; if float,
                it's the smoothing weight of a running average (higher -> smoother).
            interval_mode (str): should be one of the four options: 1) "std", 2) "minmax",
                and 3) "CI_X". The last one means confidence interval, where 'X'
                should be one of ``(80,85,90,95,99)`` indicating the confidence
                level (percentage).
        Returns:
            MeanCurve: a mean curve structure.
        """
        if not isinstance(event_file, list):
            event_file = [event_file]
        else:
            assert len(event_file) > 0, "Empty event file list!"

        ys = []
        scalar_events_list = []
        for ef in event_file:
            event_acc = EventAccumulator(ef)
            event_acc.Reload()
            # 'scalar_events' is a list of ScalarEvent(wall_time, step, value)
            scalar_events = event_acc.Scalars(self._get_metric_name())
            scalar_events_list.append(scalar_events)

        if x_steps is None:
            max_x, min_x, num_steps = int(1e15), 0, 0
            for scalar_events in scalar_events_list:
                if x_scale == 'step':
                    steps = [se.step for se in scalar_events]
                elif x_scale == 'relative':
                    t0 = scalar_events[0].wall_time
                    steps = [se.wall_time - t0 for se in scalar_events]
                else:
                    raise ValueError("Invalid x_scale value: {}".format(x_scale))
                max_x = min(max_x, steps[-1])
                min_x = max(min_x, steps[0])
                # In case we always summarize every step in the first interval,
                # len(steps) is much bigger than we expected. So we need to calculate.
                num_steps = max(
                    num_steps,
                    (steps[-1] - steps[0]) // (steps[-1] - steps[-2]))
            # calcuate x_steps by evenly dividing (min_x, max_x)
            assert max_x > min_x and num_steps > 1
            delta_x = (max_x - min_x) / (num_steps - 1)
            x_steps = np.arange(num_steps) * delta_x + min_x

        for scalar_events in scalar_events_list:
            if x_scale == 'step':
                steps, values = zip(*[(se.step, se.value) for se in scalar_events])
            elif x_scale == 'relative':
                t0 = scalar_events[0].wall_time
                steps, values = zip(*[(se.wall_time - t0, se.value) for se in scalar_events])
            else:
                raise ValueError("Invalid x_scale value: {}".format(x_scale))
            y = self._interpolate_and_smooth_if_necessary(
                steps, values, x_steps, smoothing)
            ys.append(np.array(y))

        x = x_steps
        self._mean_curve = MeanCurve.from_curves(
            x=x, ys=ys, interval_mode=interval_mode, name=name)
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self._mean_curve

    def _interpolate_and_smooth_if_necessary(self,
                                             steps,
                                             values,
                                             output_x,
                                             smoothing=None,
                                             kind="linear"):
        """First interpolate the ``(steps, values)`` pair to get a
        function. Then for the range ``(min_step, max_step)``,
        compute the values using the fitted function. Lastly apply a smoothing
        to the curve if a smoothing factor is specified.
        The reason why we have the interpolation is that
        the x steps are not always the same for multiple random runs (e.g.,
        environment steps). So we need to first adjust x steps according to
        some reference minmax steps.
        Args:
            steps (list[int]): x values
            values (list[float]): y values
            output_x (list[int]): x values for the output curve
            smoothing (int | float): if None, no smoothing is applied; if int,
                it's the window width of a Savitzky-Golay filter; if float,
                it's the smoothing weight of a running average (higher -> smoother).
            kind (str): Interpolation type. Common options: "linear" (default),
                "nearest", "cubic", "quadratic", etc. For a complete list, see
                ``scipy.interpolate.interp1d()``.
        Returns:
            tuple: the first is the adjusted x values and the second is the
                interpolated and smoothed y values.
        """
        # a rouch check to make sure the interpolation won't be too much
        assert abs(steps[-1] - output_x[-1]) / output_x[-1] < 0.2, (
            "Inconsistent final steps! actual %d output %d" % (steps[-1],
                                                               output_x[-1]))

        func = interp1d(steps, values, kind=kind, fill_value='extrapolate')
        new_values = func(output_x)

        if isinstance(smoothing, int):
            new_values = savgol_filter(new_values, smoothing, polyorder=1)
        elif smoothing is not None:
            assert 0 < smoothing < 1
            new_values = ema_smooth(new_values, weight=smoothing)

        return new_values


def ema_smooth(scalars, weight=0.6, speed=64., adaptive=False, mode="forward"):
    r"""EMA smoothing, following TB's official implementation:
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/vz_line_chart2/line-chart.ts#L695
    For adaptive EMA, the incoming weight decreases as the time increases.
    Args:
        scalars (list[float]): an array of floats to be smoothed, where the
            array index represents incoming time steps.
        weight (float): the weight of history. The history is updated as
            ``history * weight + scalar * (1 - weight)``. Only useful when
            ``adaptive=False``.
        speed (int): an integer number specifying the adpative weight. Only
            useful when ``adaptive=True``. A higher speed means a smaller
            average window.
        adaptive (bool): whether use adaptive weighting or not. If True, then
            later scalars will have smaller incoming weights (proportional to
            the inverse of array index).
        mode (str): "forward" | "both". For "forward" mode, the moving average
            goes from the array beginning to end. For "both" mode, the moving
            average has an additional backward pass, and the final smoothed
            value is an average of forward and backward passes.
    """

    def _smooth_one_pass(scalars):
        last = 0
        debias_w = 0
        smoothed = []
        w = weight
        for i, point in enumerate(scalars):
            if adaptive:
                w = 1 - speed / (i + speed)
            last = last * w + (1 - w) * point  # Calculate smoothed value
            debias_w = debias_w * w + (1 - w)
            smoothed.append(last / debias_w)

        return smoothed

    smoothed_forward = _smooth_one_pass(scalars)
    if mode != "forward":
        smoothed_backward = _smooth_one_pass(scalars[::-1])
        smoothed = np.mean(
            np.array([smoothed_forward, smoothed_backward[::-1]]), axis=0)
    else:
        smoothed = smoothed_forward
    return smoothed


class TrainAccReader(MeanCurveReader):
    """Create a mean curve reader that reads TrainAcc values."""

    def _get_metric_name(self):
        return "train/acc"

    @property
    def x_label(self):
        # return "Training Epochs"
        return "Training time (seconds)"

    @property
    def y_label(self):
        return "Training Accuracy"


class TrainLossReader(MeanCurveReader):
    """Create a mean curve reader that reads TrainLoss values."""

    def _get_metric_name(self):
        return "train/loss"

    @property
    def x_label(self):
        # return "Training Epochs"
        return "Training time (seconds)"

    @property
    def y_label(self):
        return "Training Loss"


class TestAccReader(MeanCurveReader):
    """Create a mean curve reader that reads TestAcc values."""

    def _get_metric_name(self):
        return "test/acc"

    @property
    def x_label(self):
        # return "Training Epochs"
        return "Training time (seconds)"

    @property
    def y_label(self):
        return "Test Accuracy"


class CurvesPlotter(object):
    """Plot several ``MeanCurve``s in a figure. The curve colors will form
    a cycle over 10 default colors. The user should make sure that the ``MeanCurve``s
    to plot are meaningful to be compared in one figure.
    For each ``MeanCurve``, its ``y`` field will be plotted as the mean, its
    ``min_y`` and ``max_y`` will be plotted by a shaded area around ``y``, and
    its ``x`` determines the x-axis range.
    """

    def __init__(self,
                 mean_curves,
                 y_clipping=None,
                 x_range=None,
                 y_range=None,
                 x_ticks=None,
                 x_label=None,
                 y_label=None,
                 x_scaled_and_aligned=False,
                 figsize=(4, 3),
                 dpi=100,
                 linestyle='-',
                 linewidth=2,
                 std_alpha=0.2,
                 colors=None,
                 markers=None,
                 bg_color='white',
                 grid_color='#e6e5e3',
                 plot_mean_only=False,
                 legend_kwargs=dict(loc="best"),
                 title=None):
        r"""
        Args:
            mean_curves (MeanCurve|list[MeanCurve]): each ``MeanCurve`` should
                correspond to a different method.
            x_range (tuple[float]): a tuple of ``(min_x, max_x)`` for showing on
                the figure. If None, then ``(0, 1)`` will be used. This argument is
                only used when ``x_scaled_and_aligned==True``.
            y_range (tuple[float]): a tuple of ``(min_y, max_y)`` for showing on
                the figure. If None, then it will be decided according to the
                ``y`` values. Note that this range won't change ``y`` data; it's
                only used by matplotlib for drawing ``y`` limits.
            x_ticks (list[float]): x ticks shown along x axis
            y_clipping (tuple[float]): the y values will be clipped to this range
                if not None. Because of smoothing in ``MeanCurveReader`` and/or
                std region, the input y values might be out of this range.
            x_label (str): shown besides x-axis
            y_label (str): shown besides y-axis
            x_scaled_and_aligned (bool): If True, the x axes of all ``MeanCurve``
                will be scaled and aligned so that the lower and upper :math:`x`
                bounds of all curves will be ``x_range``, and each curve's :math:`x`
                axix will be proportionally scaled. If False, the :math:`x` axis
                will be plotted according to :math:`x` of each ``MeanCurve`` as
                it is. Note that this process only involves :math:`x` scaling and
                no interpolation of :math:`y` values will ever be performed. For
                example, we have three ``MeanCurves`` to be plotted in a figure:
                .. code-block:: python
                    mean_curve1 x: (0, 100)
                    mean_curve2 x: (20, 80)
                    mean_curve3 x: (100, 200)
                with ``x_range==(0,1)``. Then in the plotted figure, the :math:`x`
                range (not x-ticks which can be specified differently!) will be
                .. code-block:: python
                    mean_curve1 x: (0, 0.5)
                    mean_curve2 x: (0.1, 0.4)
                    mean_curve3 x: (0.5, 1)
            figsize (tuple[int]): a tuple of ints determining the size of the
                figure in inches. A larger figure size will allow for longer texts,
                more axes or more ticklabels to be shown.
            dpi (int): Dots per inches. How many pixels each inch contains. A
                ``figsize`` of ``(w,h)`` consists of ``w*h*dpi**2`` pixels.
            linestyle (str|list[str]): the line style to plot. Possible values:
                '-' ('solid'), '--' ('dashed'), '-.' (dashdot), and ':' ('dotted').
                If a string, then all curves will have the same style; otherwise
                each option will apply to the corresponding curve.
            linewidth (int): the thickness of lines to plot. Default: 2.
            std_alpha (float): the transparency value for plotting shaded area around
                a curve.
            bg_color (str): the background color of the figure
            grid_color (str): color of the dashed grid lines
            plot_mean_only (bool): Whether only plot the mean curve without
                shaded regions.
            legend_kwargs (dict): kwargs for plotting the legend. If None, then
                no legend will be plotted.
            title (str): title of the figure
        """
        self._fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)

        if not isinstance(mean_curves, list):
            mean_curves = [mean_curves]

        if colors is None:
            colors = ['C%d' % i for i in range(10)]

        if markers is None:
            markers = [''] * len(mean_curves)

        if x_scaled_and_aligned:
            if x_range is None:
                x_range = (0., 1.)
            scaled_x = []
            # determine the lower and upper bounds of actual x
            min_x, max_x = int(1e15), 0
            for mc in mean_curves:
                max_x = max(max_x, mc.x[-1])
                min_x = min(min_x, mc.x[0])

            def _scale(x):
                # compute a scaled x according to the bounds
                return ((x - min_x) / (max_x - min_x) *
                        (x_range[-1] - x_range[0]) + x_range[0])

            for mc in mean_curves:
                x0, x1 = _scale(mc.x[0]), _scale(mc.x[-1])
                delta_x = (x1 - x0) / (len(mc.y) - 1)
                scaled_x.append(np.arange(len(mc.y)) * delta_x + x0)

        def _clip_y(y):
            return np.clip(y, y_clipping[0],
                           y_clipping[1]) if y_clipping else y

        if not isinstance(linestyle, list):
            linestyle = [linestyle] * len(mean_curves)
        elif len(linestyle) < len(mean_curves):
            linestyle += linestyle[-1:] * (len(mean_curves) - len(linestyle))

        for i, c in enumerate(mean_curves):
            color = colors[i % len(colors)]
            x = (scaled_x[i] if x_scaled_and_aligned else c.x)
            ax.plot(
                x,
                _clip_y(c.y),
                color=color,
                marker=markers[i],
                lw=linewidth,
                linestyle=linestyle[i],
                label=c.name)
            if not plot_mean_only:
                ax.fill_between(
                    x,
                    _clip_y(c.max_y),
                    _clip_y(c.min_y),
                    facecolor=color,
                    alpha=std_alpha)

        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        if bg_color is not None:
            ax.set_facecolor(bg_color)
        if grid_color is not None:
            ax.grid(linestyle='--', color=grid_color)
        else:
            ax.grid(linestyle='-')
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if y_range:
            ax.set_ylim(y_range)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

    def plot(self, output_path, dpi=200, transparent=False, close_fig=True):
        """Plot curves and save the figure to disk.
        Args:
            output_path (str): the output file path
            dpi (int): dpi for the figure. A higher value results in higher
                resolution.
            transparent (bool): If True, then the figure has a transparent
                background.
            close_fig (bool): whether to close/release this figure after plotting.
                If ``False``, the user has to close it manually.
        """
        self._fig.savefig(
            output_path, dpi=dpi, transparent=transparent, bbox_inches='tight')
        if close_fig:
            plt.close(self._fig)


def _get_curve_path(dataset, method, seed=0):

    # ours vs baseline
    return os.path.join(os.getenv("HOME"), 
                        "Workspace/KFAC-Pytorch/runs/pretrain", 
                        dataset, "vgg11_bn/plot", method, "seed%d" % (seed))

    # # baseline abblation
    # return os.path.join(os.getenv("HOME"), 
    #                     "Workspace/KFAC-Pytorch/runs/pretrain", 
    #                     dataset, "vgg11_bn/plot", method)


if __name__ == "__main__":
    """Plotting examples."""
    # ours vs baseline
    methods = ["baseline", "ours"]
    seeds = [0, 1, 2, 3]
    # dataset = "fashion_mnist"
    dataset = "cifar10"
    # curve_name = "testacc"
    # curve_name = "trainacc"
    curve_name = "trainloss"

    curve_readers = [[
        # TestAccReader(
        # TrainAccReader(
        TrainLossReader(
            event_file=[_get_curve_path(dataset, method, seed) for seed in seeds],
            # x_steps=np.arange(0, 50),
            x_scale='relative',
            name="%s" % (method)) 
    ] for method in methods]

    # Scale and align x-axis of SAC and DDPG on task "kickball"
    plotter = CurvesPlotter([cr[0]() for cr in curve_readers],
                            x_label=curve_readers[0][0].x_label,
                            y_label=curve_readers[0][0].y_label,
                            # y_range=(0, 100),
                            x_range=(0, 50),
                            linewidth=2)
    plotter.plot(output_path="./plots/rebuttal/%s_%s.png" % (dataset, curve_name))


    # # abblation for baseline
    # methods = ["baseline", "baseline_lr", "baseline_scale", "baseline_lr_scale"]

    # dataset = "fashion_mnist"
    # # dataset = "cifar100"
    # curve_name = "testacc"
    # # curve_name = "trainacc"
    # # curve_name = "trainloss"

    # curve_readers = [[
    #     TestAccReader(
    #     # TrainAccReader(
    #     # TrainLossReader(
    #         event_file=[_get_curve_path(dataset, method)],
    #         # x_steps=np.arange(0, 50),
    #         name="%s" % (method)) 
    # ] for method in methods]

    # # Scale and align x-axis of SAC and DDPG on task "kickball"
    # plotter = CurvesPlotter([cr[0]() for cr in curve_readers],
    #                         x_label=curve_readers[0][0].x_label,
    #                         y_label=curve_readers[0][0].y_label,
    #                         # y_range=(0, 100),
    #                         x_range=(0, 50),
    #                         linewidth=2)
    # plotter.plot(output_path="./plots/baseline_%s_%s.png" % (dataset, curve_name))
