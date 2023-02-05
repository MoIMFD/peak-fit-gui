import argparse

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import RadioButtons, RangeSlider, Slider, Button
from matplotlib.patches import Polygon


def set_polygon_y(polygon: Polygon, y_lower: float, y_upper: float):
    coordinate_array = polygon.get_xy()
    coordinate_array[:, 1] = [y_lower, y_upper, y_lower, y_upper]
    polygon.set_xy(coordinate_array)


class CLIObject:
    def __init__(self) -> None:
        pass


class MatplotlibGUI:
    gui_title = "scipy.signal.find_peaks - GUI"
    line_plot_params = dict()
    pos_scatter_plot_params = dict(
        ls="None",
        marker="s",
        markeredgecolor="red",
        markerfacecolor="None",
    )
    neg_scatter_plot_params = pos_scatter_plot_params.copy()
    neg_scatter_plot_params["markeredgecolor"] = "blue"
    hline_params = dict(ls="--", color="black", lw=1)

    def redraw(method):
        def wrapper(self, *args, **kwargs):
            method(self, *args, **kwargs)
            self.figure.canvas.draw_idle()

        return wrapper

    def __init__(self, df, find_peaks_wrapper, x_index=0, y_index=1) -> None:
        self.df = df
        self.find_peaks_wrapper = find_peaks_wrapper
        ###########################
        ## setup figure and axes ##
        ###########################
        self.figure, self.axes = plt.subplots(figsize=(16, 9))
        self.figure.subplots_adjust(left=0.3, bottom=0.25)
        self.figure.suptitle(self.gui_title, fontsize=16)
        self.grid_flag = False
        ###########################
        ## setup x radio buttons ##
        ###########################
        self.x_data_radio = RadioButtons(
            self.figure.add_axes(
                [
                    0.02,
                    self.axes.get_position().x0 / 2,
                    0.15,
                    0.05 * self.df.columns.size,
                ],
                aspect="equal",
            ),
            self.df.columns,
            active=x_index,
            activecolor="blue",
        )
        self.x_data_radio.on_clicked(self.x_data_radio_function)
        self.axes.set_xlabel("x Axis", color=self.x_data_radio.activecolor)
        ###########################
        ## setup y radio buttons ##
        ###########################
        self.y_data_radio = RadioButtons(
            self.figure.add_axes(
                [
                    0.02,
                    self.axes.get_position().x1 / 2,
                    0.15,
                    0.05 * self.df.columns.size,
                ],
                aspect="equal",
            ),
            self.df.columns,
            active=y_index,
            activecolor="red",
        )
        self.y_data_radio.on_clicked(self.y_data_radio_function)
        self.axes.set_ylabel("y Axis", color=self.y_data_radio.activecolor)
        ##############################
        ## create initial line plot ##
        ##############################
        (self.line,) = self.axes.plot(
            self.xdata, self.ydata, **self.line_plot_params
        )
        ####################################
        ## create height parameter slider ##
        ####################################
        self.height_slider = RangeSlider(
            self.figure.add_axes(
                [
                    self.axes.get_position().x1 + 0.03,
                    self.axes.get_position().y0,
                    0.05 * 9 / 16,
                    self.axes.get_position().height,
                ]
            ),
            "height",
            orientation="vertical",
            valmin=self.axes.get_ylim()[0],
            valmax=self.axes.get_ylim()[1],
            valinit=(min(self.line.get_ydata()), max(self.line.get_ydata())),
        )
        self.height_slider.on_changed(self.height_slider_function)
        self.hline_upper = self.axes.axhline(
            self.height_slider.val[1], **self.hline_params
        )
        self.hline_lower = self.axes.axhline(
            self.height_slider.val[0], **self.hline_params
        )
        # self.hspan_lower = self.axes.axhspan(
        #     ymin=self.axes.get_ylim()[0], ymax=self.hline_lower.get_ydata()[0]
        # )
        # self.hspan_upper = self.axes.axhspan(
        #     ymin=self.hline_upper.get_ydata()[0], ymax=self.axes.get_ylim()[1]
        # )
        ######################################
        ## create distance parameter slider ##
        ######################################
        self.distance_slider = Slider(
            self.figure.add_axes(
                [
                    self.axes.get_position().x0,
                    self.axes.get_position().y0 - 0.1,
                    self.axes.get_position().width,
                    0.05,
                ]
            ),
            "distance",
            orientation="horizontal",
            valmin=0,
            valmax=self.xdata.size,
            valinit=0,
            valstep=int(1),
        )
        self.distance_slider.on_changed(self.distance_slider_function)
        ########################################
        ## create prominence parameter slider ##
        ########################################
        self.prominence_slider = RangeSlider(
            self.figure.add_axes(
                [
                    self.axes.get_position().x0,
                    self.axes.get_position().y0 - 0.15,
                    self.axes.get_position().width,
                    0.05,
                ]
            ),
            "prominence",
            orientation="horizontal",
            valmin=0,
            valmax=abs(self.ydata.max() - self.ydata.min()),
            valinit=(0, abs(self.ydata.max() - self.ydata.min())),
        )
        self.prominence_slider.on_changed(self.prominence_slider_function)
        ###################################
        ## create width parameter slider ##
        ###################################
        self.width_slider = RangeSlider(
            self.figure.add_axes(
                [
                    self.axes.get_position().x0,
                    self.axes.get_position().y0 - 0.2,
                    self.axes.get_position().width,
                    0.05,
                ]
            ),
            "width",
            orientation="horizontal",
            valmin=0,
            valmax=self.xdata.size + 1,
            valinit=(0, self.xdata.size + 1),
            valstep=int(1),
        )
        self.width_slider.on_changed(self.width_slider_function)
        #################################
        ## create initial scatter plot ##
        #################################
        (self.pos_scatter,) = self.axes.plot(
            self.xdata[self.find_peaks_wrapper.get_pos_peaks(self.ydata)[0]],
            self.ydata[self.find_peaks_wrapper.get_pos_peaks(self.ydata)[0]],
            **self.pos_scatter_plot_params
        )
        (self.neg_scatter,) = self.axes.plot(
            self.xdata[self.find_peaks_wrapper.get_neg_peaks(self.ydata)[0]],
            self.ydata[self.find_peaks_wrapper.get_neg_peaks(self.ydata)[0]],
            **self.neg_scatter_plot_params
        )
        #########################
        ## create print button ##
        #########################
        print_button = Button(
            self.figure.add_axes([0.025, 0.14, 0.1, 0.04]),
            "Print",
            hovercolor="0.975",
        )
        print_button.on_clicked(self.print_btn_function)
        ########################
        ## create grid button ##
        ########################
        grid_button = Button(
            self.figure.add_axes([0.025, 0.02, 0.1, 0.04]),
            "Grid",
            hovercolor="0.975",
        )
        grid_button.on_clicked(self.grid_btn_function)

        plt.show()

    @property
    def xdata(self):
        return self.df.iloc[:, self.x_index].values

    @property
    def ydata(self):
        return self.df.iloc[:, self.y_index].values

    @property
    def x_index(self):
        return self.df.columns.get_loc(self.x_data_radio.value_selected)

    @property
    def y_index(self):
        return self.df.columns.get_loc(self.y_data_radio.value_selected)

    @redraw
    def x_data_radio_function(self, event):
        self.set_line()
        self.set_scatter()
        self.axes.relim()
        self.axes.autoscale_view()

    @redraw
    def y_data_radio_function(self, event):
        self.set_line()
        self.set_scatter()
        self.height_slider.val = (
            self.ydata.min(),
            self.ydata.max(),
        )

        self.set_hlines(*self.height_slider.val)
        self.axes.relim()
        self.axes.autoscale_view()
        self.height_slider.ax.set_ylim(*self.axes.get_ylim())
        self.height_slider.valmin = self.axes.get_ylim()[0]
        self.height_slider.valmax = self.axes.get_ylim()[1]

    def set_line(self):
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)

    def set_scatter(self):
        self.pos_scatter.set_xdata(
            self.xdata[self.find_peaks_wrapper.get_pos_peaks(self.ydata)[0]]
        )
        self.pos_scatter.set_ydata(
            self.ydata[self.find_peaks_wrapper.get_pos_peaks(self.ydata)[0]]
        )
        self.neg_scatter.set_xdata(
            self.xdata[self.find_peaks_wrapper.get_neg_peaks(self.ydata)[0]]
        )
        self.neg_scatter.set_ydata(
            self.ydata[self.find_peaks_wrapper.get_neg_peaks(self.ydata)[0]]
        )

    def set_hlines(self, *vals):
        self.hline_lower.set_ydata(vals[0])
        self.hline_upper.set_ydata(vals[1])

    @redraw
    def height_slider_function(self, event):
        self.find_peaks_wrapper.update_param("height", event)
        self.set_hlines(*event)
        self.set_scatter()

    @redraw
    def distance_slider_function(self, event):
        if event == 0:
            self.find_peaks_wrapper.update_param("distance", None)
        else:
            self.find_peaks_wrapper.update_param("distance", event)
        self.set_scatter()

    @redraw
    def width_slider_function(self, event):
        if event[0] == self.width_slider.valmin:
            event = (None, event[1])
        if event[1] > self.width_slider.valmax:
            event = (event[0], None)
        self.find_peaks_wrapper.update_param("width", event)
        self.set_scatter()

    @redraw
    def prominence_slider_function(self, event):
        if event[0] == self.prominence_slider.valmin:
            event = (None, event[1])
        if event[1] == self.prominence_slider.valmax:
            event = (event[0], None)
        self.find_peaks_wrapper.update_param("prominence", event)
        self.set_scatter()

    @redraw
    def grid_btn_function(self, event):
        self.grid_flag = not self.grid_flag
        self.axes.grid(self.grid_flag)

    def print_btn_function(self, event):
        self.find_peaks_wrapper.print_peaks()
