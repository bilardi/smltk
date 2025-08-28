"""The class for managing the data of the main repositories

A collection of methods to simplify your code.
"""

import datetime
import matplotlib.dates as mpld
import matplotlib.pyplot as plt
import numpy as np


class Indicator:
    """
    The class Indicator contains the tool kit to calculate the principal indicators.

    Arguments: params (dict) with the keys below
        :events (list[str]): list of directional change events
        :timeseries (list[int|float]): list of values, default None

    Here's an example:

        >>> from smltk.feature_engineering import Indicator
        >>> timeseries = numpy.array()
        >>> indicator = Indicator()
        >>> dc_events = indicator.get_dc_events(timeseries)
        >>> print(dc_events)
        array['upward dc', 'downward dc', ..]
    """

    events = None
    timeseries = None

    def __init__(self, params={}):
        if "events" in params:
            self.events = params["events"]
        if "timeseries" in params:
            self.timeseries = params["timeseries"]

    def get_dc_events(
        self, timeseries: np.array = None, threshold: float = 0.0001
    ) -> list:
        """
        Compute all relevant Directional Change parameters

        Arguments:
            :timeseries (list[int|float]): list of values
            :threshold (float): default is 0.0001
        Returns:
            list of directional change events
        """
        if timeseries is None and self.timeseries is None:
            raise ValueError(
                "Timeseries data has to be a no empty numpy.array()"
            )
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries

        time_value_list = []
        time_point_list = []
        events = []

        ext_point_n = timeseries[0]
        curr_event_max = timeseries[0]
        curr_event_min = timeseries[0]
        time_point_max = 0
        time_point_min = 0
        trend_status = "up"
        time_point = 0
        for i, ts_value in enumerate(timeseries):
            time_value = (ts_value - ext_point_n) / (ext_point_n * threshold)
            time_value_list.append(time_value)
            time_point_list.append(time_point)
            time_point += 1
            if trend_status == "up":
                events.append("upward overshoot")
                if ts_value < ((1 - threshold) * curr_event_max):
                    trend_status = "down"
                    curr_event_min = ts_value
                    ext_point_n = curr_event_max
                    time_point = i - time_point_max
                    num_points_change = i - time_point_max
                    for j in range(1, num_points_change + 1):
                        events[-j] = "downward dc"
                else:
                    if ts_value > curr_event_max:
                        curr_event_max = ts_value
                        time_point_max = i
            else:
                events.append("downward overshoot")
                if ts_value > ((1 + threshold) * curr_event_min):
                    trend_status = "up"
                    curr_event_max = ts_value
                    ext_point_n = curr_event_min
                    time_point = i - time_point_min
                    num_points_change = i - time_point_min
                    for j in range(1, num_points_change + 1):
                        events[-j] = "upward dc"
                else:
                    if ts_value < curr_event_min:
                        curr_event_min = ts_value
                        time_point_min = i
        self.events = events
        return events

    def get_dc_events_starts(
        self, events: list = None, timeseries: list = None
    ) -> dict:
        """
        Get only Directional Changes starts

        Arguments:
            :events (list[str]): list of directional change events
            :timeseries (list[int|float]): list of values
        Returns:
            dictionary of boolean lists when each directional change events starts
        """
        starts = {}
        previous_change = None
        if events is None and self.events is None:
            raise ValueError("Events data has to be a no empty numpy.array()")
        if events is None and self.events is not None:
            events = self.events
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries
        directional_changes = set(events)
        for directional_change in directional_changes:
            if directional_change not in starts:
                starts[directional_change] = []
        for index, current_change in enumerate(events):
            for directional_change in directional_changes:
                starts[directional_change].append(0)
            if previous_change != current_change:
                starts[current_change][-1] = (
                    1 if timeseries is None else timeseries[index]
                )
            previous_change = current_change
        return starts

    def get_dc_events_ends(
        self, events: list = None, timeseries: list = None
    ) -> dict:
        """
        Get only Directional Changes ends

        Arguments:
            :events (list[str]): list of directional change events
            :timeseries (list[int|float]): list of values
        Returns:
            dictionary of boolean lists when each directional change events ends
        """
        ends = {}
        previous_change = None
        if events is None and self.events is None:
            raise ValueError("Events data has to be a no empty numpy.array()")
        if events is None and self.events is not None:
            events = self.events
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries
        directional_changes = set(events)
        for directional_change in directional_changes:
            if directional_change not in ends:
                ends[directional_change] = []
        for index, current_change in enumerate(events):
            for directional_change in directional_changes:
                ends[directional_change].append(0)
            if previous_change != current_change:
                if previous_change is not None:
                    ends[previous_change][-2] = (
                        1 if timeseries is None else timeseries[index]
                    )
            previous_change = current_change
        ends[previous_change][-1] = 1 if timeseries is None else timeseries[-1]
        return ends

    def plot_dc(self, params: dict = {}, return_ax=False):
        """
        Plot image with directional changes

        Arguments: params (dict) with the keys below
            :dc_colors (dict): key-value about each event-color of directional change
            :events (list[str]): list of events names for time point
            :timeseries (list[float]): list of values
            :timestamp (list[int|datetime]): time point list
            :figsize (tuple): default (10, 5)
            :title (str): title of plot
            :x_axis_label (str): label of x axis
            :y_axis_label (str): label of y axis

        Returns:
            plot or its object
        """
        if "dc_colors" not in params:
            params["dc_colors"] = {
                "upward dc": "green",
                "upward overshoot": "lime",
                "downward dc": "red",
                "downward overshoot": "lightcoral",
            }
        if "figsize" not in params:
            params["figsize"] = (10, 5)
        if (
            "timeseries" in params
            and "timestamp" in params
            and "events" in params
        ):
            _, ax1 = plt.subplots(figsize=params["figsize"])
            ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
            if isinstance(params["timestamp"][0], type(datetime.datetime)):
                dates = mpld.date2num(params["timestamp"])
                ax1.plot_date(
                    dates, params["timeseries"], color="white", markersize=0.1
                )
            handles = []
            for event in set(params["events"]):
                color = params["dc_colors"][event]
                handles.append(
                    plt.Line2D(
                        [params["timestamp"][0]], [0], color=color, label=event
                    )
                )
            for index, event in enumerate(params["events"]):
                ax1.plot(
                    params["timestamp"][index : index + 2],
                    params["timeseries"][index : index + 2],
                    color=params["dc_colors"][event],
                )
            if "title" in params:
                ax1.set_title(params["title"])
            if "x_axis_label" in params:
                ax1.set_xlabel(params["x_axis_label"])
            if "y_axis_label" in params:
                ax1.set_ylabel(params["y_axis_label"])
            ax1.legend(handles=handles, loc="upper right", fontsize="small")
            if return_ax is True:
                return ax1
            plt.show()
