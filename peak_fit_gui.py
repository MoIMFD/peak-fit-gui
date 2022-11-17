import sys

# check for python >= 3.11 
if sys.version_info[1] >= 11:
    import tomllib
else:
    import tomli as tomllib

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.signal import find_peaks

### GLOBALS ###
SCRIPT_VERSION:str = "1.0"  # current version of the script, update this if you make significant modifications

AUTHOR:str = "Moritz Kluwe"  # original script author
CREATED_AT:str = "16.11.2022"  # date of script creation

MODIFIED_BY:str = []  # list of people how contribute major modifications; change SCRIPT_VERSION variable
MODIFIED_AT:str = []  # dates at which major modifications have taken place

CONSOLE_OUTPUT_PADDING:int = 14  # padding for column wise output of the 'print' button
###############


def set_polygon_y(polygon: Polygon, y_lower:float, y_upper:float):
    """Helper function for updating lower and upper of a matplotlib.patches.Polygon in square shape, e. g. a box.
    (x0, y_upper) --- (x1, y_upper)
          |                 |
          |                 |
    (x0, y_lower) --- (y1, y_lower)

    Parameters
    ----------
    polygon : Polygon
        matplotlib.patches.Polygon with square shape like returned by matplotlib.pyplot.axhspan
    y_lower : float
        new lower y coordinate value
    y_upper : float
        new upper y coordinate value
    """
    _array = polygon.get_xy()
    _array[:, 1] = [y_lower, y_upper, y_upper, y_lower, y_lower]
    polygon.set_xy(_array)

class PeakFitGUI:
    def __init__(
        self,
        df :pd.DataFrame,
        plot_title: str = "", 
        x_initial_index :int=1,
        y_initial_index :int=2,
        peak_fit_defaults :dict = dict(),
        slider_limit_defaults:dict = dict(),
        radio_colors:dict = dict(),
        figure_settings:dict = dict(),
        plot_settings:dict = dict(),
        scatter_pos_peaks_params:dict = dict(),
        scatter_neg_peaks_params:dict = dict(),
        hline_settings:dict = dict(),
        axhspan_settings:dict = dict(),
        *args,
        **kwargs
        ) -> None:
        # data frame and index definitions
        self.df = df
        self.ncols = len(self.df.columns)
        self.df.insert(loc=0, column="data_index", value=df.index)  # create a new column based on data index
        # set initial indexes for x and y data
        self.idx = x_initial_index
        self.idy = y_initial_index
        # initialize matplotlib figure and axes
        self.fig, self.ax = plt.subplots(**figure_settings)
        self.fig.suptitle("Peak Fit GUI", fontsize=16)
        self.ax.set_title(plot_title)
        self.grid_visible = False
        self.line, *_ = self.ax.plot(
            self.xdata, 
            self.ydata, 
            **plot_settings
            ) # when 'dict.get' method is called with a key not present in the dict it returns None
        self.ax.set_xlabel(
            df.columns[self.idx],
            color=radio_colors["x_color"] if radio_colors.get("x_color") is not None else "black"
            )
        self.ax.set_ylabel(
            df.columns[self.idy],
            color=radio_colors["y_color"] if radio_colors.get("y_color") is not None else "black"
            )
        ## create radio buttons for index selection
        self.fig.subplots_adjust(left=0.3)
        ## BEGIN idx_radio
        self.idx_radio_axes = self.fig.add_axes([0.02, 0.45 - 0.03*self.ncols, 0.15, 0.03*self.ncols])
        self.idx_radio_axes.set_title("x data")
        self.idx_radio = RadioButtons(
            self.idx_radio_axes, 
            df.columns, active=self.idx, 
            activecolor=radio_colors["x_color"] if radio_colors.get("x_color") is not None else "black"
            )
        self.idx_radio.on_clicked(self.radio_idx_function)
        ## END idx_radio
        ## BEGIN idy_radio
        self.idy_radio_axes = self.fig.add_axes([0.02, 0.6, 0.15, 0.03*self.ncols])
        self.idy_radio_axes.set_title("y data")
        self.idy_radio = RadioButtons(
            self.idy_radio_axes, 
            df.columns, active=self.idy, 
            activecolor = radio_colors["y_color"] if radio_colors.get("y_color") is not None else "black"
            )
        self.idy_radio.on_clicked(self.radio_idy_function)
        ## END idy_radio
        ## create slider for parameter variation
        self.fig.subplots_adjust(bottom=0.25)
        ## BEGIN height_slider
        self.height_min, self.height_max = peak_fit_defaults.get("height") if peak_fit_defaults.get("height") is not None else (self.ydata.min(), self.ydata.max())
        # minimum slider
        self.height_min_slider_axes = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])
        self.height_min_slider = Slider(
            self.height_min_slider_axes, 
            "min height", 
            valmin =  self.ydata.min(),
            valmax = self.height_max, 
            valinit = self.height_min
             )
        self.height_min_slider.on_changed(self.height_min_function)
        self.height_min_line = self.ax.axhline(self.height_min, **hline_settings)
        self.height_min_box = self.ax.axhspan(
            ymin = self.ylim[0],
            ymax = self.height_min,
            **axhspan_settings
        )
        # maximum slider
        self.height_max_slider_axes = self.fig.add_axes([0.25, 0.12, 0.65, 0.03])
        self.height_max_slider = Slider(
            self.height_max_slider_axes, 
            "max height", 
            valmin = self.height_min, 
            valmax = self.ydata.max(), 
            valinit = self.height_max
            )
        self.height_max_slider.on_changed(self.height_max_function)
        self.height_max_line = self.ax.axhline(self.height_max, **hline_settings)
        self.height_max_box = self.ax.axhspan(
            ymin = self.height_max,
            ymax = self.ylim[1],
            **axhspan_settings
        )
        ## END height_slider
        ## BEGIN distance_slider
        self.peak_distance = peak_fit_defaults["distance"] if peak_fit_defaults.get("distance") is not None else 1
        self.peak_distance_slider_axes = self.fig.add_axes([0.25, 0.09, 0.65, 0.03])
        self.peak_distance_slider = Slider(
            self.peak_distance_slider_axes, 
            "distance", 
            valmin = slider_limit_defaults["distance_lower"] if slider_limit_defaults.get("distance_lower") is not None else 1,
            valmax = slider_limit_defaults["distance_upper"] if slider_limit_defaults.get("distance_upper") is not None else self.xdata.size,
            valinit = self.peak_distance, 
            valstep = 1
            )
        self.peak_distance_slider.on_changed(self.peak_distance_function)
        ## END distance_slider
        ## BEGIN prominence_slider
        self.prominence_min, self.prominence_max = peak_fit_defaults["prominence"] if peak_fit_defaults.get("prominence") is not None else (0, 1)
        # minimum slider
        self.prominence_min_slider_axes = self.fig.add_axes([0.25, 0.06, 0.65, 0.03])
        self.prominence_min_slider = Slider(
            self.prominence_min_slider_axes,
            "prom. min",
            valmin = slider_limit_defaults["prominence_lower"] if slider_limit_defaults.get("prominence_lower") is not None else 0,
            valmax = self.prominence_max,
            valinit = self.prominence_min
            )
        self.prominence_min_slider.on_changed(self.prominence_min_function)
        # maximum slider
        self.prominence_max_slider_axes = self.fig.add_axes([0.25, 0.03, 0.65, 0.03])
        self.prominence_max_slider = Slider(
            self.prominence_max_slider_axes,
            "prom. max",
            valmin=self.prominence_min,
            valmax=slider_limit_defaults["prominence_upper"] if slider_limit_defaults.get("prominence_upper") is not None else 1,
            valinit=self.prominence_max
            )
        self.prominence_max_slider.on_changed(self.prominence_max_function)
        ## END prominence_slider
        ## BEGIN print_button
        self.print_button_axes = plt.axes([0.025, 0.13, 0.1, 0.04])
        self.print_button = Button(self.print_button_axes, "Print", hovercolor="0.975")
        self.print_button.on_clicked(self.print_peaks)
        ## END print_button
        ## BEGIN reset_button
        self.reset_button_axes = plt.axes([0.025, 0.07, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_axes, "Reset", hovercolor="0.975")
        self.reset_button.on_clicked(self.reset_params)
        ## END reset_button
        ## BEGIN grid_button
        self.grid_button_axes = plt.axes([0.025, 0.01, 0.1, 0.04])
        self.grid_button = Button(self.grid_button_axes, "Grid", hovercolor="0.975")
        self.grid_button.on_clicked(self.switch_grid)
        ## END grid_button
        # scatter plot for positive peaks 
        (self.scatter_pos_peaks,) = self.ax.plot(
            self.xdata[self.pos_peaks[0]],
            self.ydata[self.pos_peaks[0]],
            **scatter_pos_peaks_params
        )
        # scatter plot for negative peaks
        (self.scatter_neg_peaks,) = self.ax.plot(
            self.xdata[self.neg_peaks[0]],
            self.ydata[self.neg_peaks[0]],
            **scatter_neg_peaks_params
        )

        self.ax.legend()
        self.ax.set_ylim(self.ylim)

    @property
    def param_dict(self) -> dict:
        """Returns parameter dictionary used in 'scipy.signal.find_peaks' based on current slider values

        Returns
        -------
        dict
            parameter dict for 'scipy.signal.find_peaks'
        """
        return {
            "height": (self.height_min, self.height_max),
            "distance": self.peak_distance,
            "prominence": (
                None if self.prominence_min <= 0 else self.prominence_min,
                None if self.prominence_max == self.prominence_min_slider.valmax else self.prominence_max
                ),
            "width": (0, 20),
        }

    @property
    def param_dict_neg(self) -> dict:   
        """Calls self.parameter_dict and reverses the "height" bounds for usage with negative signal        

        Returns
        -------
        dict
            self.parameter_dict with reversed "height" parameter
        """
        param_dict = self.param_dict.copy()
        param_dict["height"] = (-self.height_max, -self.height_min)
        return param_dict

    @property
    def ydata(self) -> pd.Series:
        """Returns column specified by 'self.idy' from 'self.df'

        Returns
        -------
        pd.Series
            current y data specified by 'self.idy'
        """
        return self.df.iloc[:, self.idy]

    @property
    def xdata(self) -> pd.Series:
        """Returns column specified by 'self.idx' from 'self.df'

        Returns
        -------
        pd.Series
            current x data specified by 'self.idx'
        """
        return self.df.iloc[:, self.idx]
        
    @property
    def ylim(self) -> tuple[float, float]:
        """Defines the y limits of the plot axes ('self.ax') based on which is greater / smaller data or height parameter.
        Limits are extended by 1/10 of the data range for nicer representation. 

        Returns
        -------
        tuple[float, float]
            _description_
        """
        ylim_lower = min(self.ydata.min(), self.height_min)
        ylim_upper = max(self.ydata.max(), self.height_max)
        delta = ylim_upper - ylim_lower
        return (
            ylim_lower - delta/10,
            ylim_upper + delta/10
        )

    def reset_slider_axes(self):
        """Resets the axes of all sliders based on the current slider upper and lower limits
        """
        self.height_min_slider_axes.set_xlim(
            self.height_min_slider.valmin,
            self.height_min_slider.valmax
            )
        self.height_max_slider_axes.set_xlim(
            self.height_max_slider.valmin,
            self.height_max_slider.valmax
            )

        self.prominence_min_slider_axes.set_xlim(
            self.prominence_min_slider.valmin,
            self.prominence_min_slider.valmax
        )
        self.prominence_max_slider_axes.set_xlim(
            self.prominence_max_slider.valmin,
            self.prominence_max_slider.valmax
        )

    def radio_idx_function(self, label:str):
        """Evenet function for interaction with radio_idx widget. Restes 'self.idx' to radio selection and changes axes limits and
        labels accordingly.

        Parameters
        ----------
        label : str
            string containing the selected label of the radio widget
        """
        self.idx = self.df.columns.get_loc(label)
        # update line data
        self.line.set_xdata(self.xdata)
        # update x axis
        self.ax.set_xlim(self.xdata.min(), self.xdata.max())
        self.ax.set_xlabel(label)
        self.update_peak_scatter()
        plt.draw()


    def radio_idy_function(self, label:str):
        """Evenet function for interaction with radio_idy widget. Restes 'self.idy' to radio selection and changes axes limits and
        labels accordingly. Additionally resets the height slider bounds to current data range.

        Parameters
        ----------
        label : str
            string containing the selected label of the radio widget
        """
        self.idy = self.df.columns.get_loc(label)
        # reset height slider bounds
        self.height_max_slider.valmax = self.ydata.max()
        self.height_min_slider.valmin = self.ydata.min()
        self.reset_slider_axes()
        # update line data
        self.line.set_ydata(self.ydata)
        # update y axis
        self.ax.set_ylim(self.ylim)
        self.ax.set_ylabel(label)
        self.update_peak_scatter()
        plt.draw()

    def height_min_function(self, event:str):
        """Event function for height min slider. Reads slider value, resets maximum slider minimum to slider value, updates slider axes
        and plot.

        Parameters
        ----------
        event : str
            slider event
        """
        self.height_min = self.height_min_slider.val
        self.height_max_slider.valmin = self.height_min
        self.reset_slider_axes()
        self.update_peak_scatter()
        # reset horizontal line
        self.height_min_line.set_ydata(self.height_min)
        # reset out of range box
        set_polygon_y(
            polygon=self.height_min_box,
            y_lower=self.ylim[0],
            y_upper=self.height_min
            )
        self.ax.set_ylim(self.ylim)
        plt.draw()

    def height_max_function(self, event):
        """Event function for height max slider. Reads slider value, resets minimum slider maximum to slider value, updates slider axes
        and plot.

        Parameters
        ----------
        event : str
            slider event
        """
        self.height_max = self.height_max_slider.val
        self.height_min_slider.valmax = self.height_max
        self.reset_slider_axes()
        self.update_peak_scatter()
        # reset horizontal line
        self.height_max_line.set_ydata(self.height_max)
        # reset out of range box
        set_polygon_y(
            polygon=self.height_max_box,
            y_lower=self.height_max,
            y_upper=self.ylim[1]
            )
        self.ax.set_ylim(self.ylim)
        plt.draw()

    def peak_distance_function(self, event):
        """Event function for peak distance slider.

        Parameters
        ----------
        event : _type_
            slider event
        """
        self.peak_distance = self.peak_distance_slider.val
        self.update_peak_scatter()
        plt.draw()

    def prominence_min_function(self, event):
        """Event function for prominence min slider

        Parameters
        ----------
        event : _type_
            slider event
        """
        self.prominence_min = self.prominence_min_slider.val
        self.prominence_max_slider.valmin = self.prominence_min
        self.reset_slider_axes()
        self.update_peak_scatter()
        plt.draw()

    def prominence_max_function(self, event):
        """Event function for prominence max slider

        Parameters
        ----------
        event : _type_
            slider event
        """
        self.prominence_max = self.prominence_max_slider.val
        self.prominence_min_slider.valmax = self.prominence_max
        self.reset_slider_axes()
        self.update_peak_scatter()
        plt.draw()

    @property
    def neg_peaks(self):# -> tuple[np.ndarray | dict]:
        """Returns peaks found with current parameter settings for negative signal using self.paramter_dict_neg

        Returns
        -------
        tuple[np.ndarray | dict]
            array of peaks indexes and dictionary containing information on peaks
        """
        return find_peaks(-self.ydata, **self.param_dict_neg)

    @property
    def pos_peaks(self):# -> tuple[np.ndarray | dict]:
        """Returns peaks found with current parameter settings for positive signal using self.paramter_dict

        Returns
        -------
        tuple[np.ndarray | dict]
            array of peaks indexes and dictionary containing information on peaks
        """
        return find_peaks(self.ydata, **self.param_dict)

    def update_peak_scatter(self):
        """Updates the peak scatter plots for positive and negative peaks
        """
        self.scatter_pos_peaks.set_data(self.xdata[self.pos_peaks[0]], self.ydata[self.pos_peaks[0]])
        self.scatter_neg_peaks.set_data(self.xdata[self.neg_peaks[0]], self.ydata[self.neg_peaks[0]])


    def reset_params(self, event):
        """Event function for reset_button. Resets all sliders to initial value

        Parameters
        ----------
        event : _type_
            button event
        """
        self.height_min_slider.reset()
        self.height_max_slider.reset()
        self.peak_distance_slider.reset()
        self.prominence_min_slider.reset()
        self.prominence_max_slider.reset()

    def switch_grid(self, event):
        self.grid_visible = ~self.grid_visible
        self.ax.grid(self.grid_visible)
        plt.draw()

    def print_peaks(self, event):
        """Event function for print_button. Prints current parameter settings and peaks with all their information to console 

        Parameters
        ----------
        event : _type_
            button event
        """
        padding = CONSOLE_OUTPUT_PADDING
        # print names of active x and y columns
        print(f"x column = {self.df.columns[self.idx]}; y column = {self.df.columns[self.idy]}")
        ## BEGIN processing positive peaks
        print(f"+ Positive Peaks ({self.pos_peaks[0].size:d}) +")  # number of peaks
        # print current parameter settings for positive peaks
        print(
            ", ".join([f"{key}: {value}" for key, value in self.param_dict.items()])
            + "\n"
        )
        # create list of column headers
        headers = ["no", "peak index", "x value", "y value"]
        # append list of evaluated peak properties
        headers.extend(self.pos_peaks[1].keys())
        # print headers padded with padding length
        print("".join(head.rjust(padding) for head in headers))
        # loop for printing row values for each column
        for i, values in enumerate(
            zip(
                self.pos_peaks[0],  # peak indexes
                self.xdata[self.pos_peaks[0]],  # x values ar peak indexes
                self.ydata[self.pos_peaks[0]],  # 
                *self.pos_peaks[1].values()  # evaluated peak parameters
                )
            ):
            # print row
            print(f"{i:{padding}d}" + "".join(f"{val:{padding}g}" for val in values))
        ## END processing positive peaks
        ## BEGIN processing negative peaks
        print(f"\n- Negative Peaks ({self.neg_peaks[0].size:d}) -")  # number of peaks
        # print current parameter settings for negative peaks 
        print(
            ", ".join([f"{key}: {value}" for key, value in self.param_dict_neg.items()])
            + "\n"
        )
        # create list of column headers
        headers = ["no", "peak index", "x value", "y value"]
        # append list of evaluated peak properties
        headers.extend(self.neg_peaks[1].keys())
        # print headers padded with padding length
        print("".join(head.rjust(padding) for head in headers))
        # loop for printing row values for each column
        for i, values in enumerate(
            zip(
                self.neg_peaks[0],  # peak indexes
                self.xdata[self.neg_peaks[0]],  # x values at peak indexes
                self.ydata[self.neg_peaks[0]], # y values at peak indexes
                *self.neg_peaks[1].values()  # evaluated peak parameters
                )
            ):
            # print row
            print(f"{i:{padding}d}" + "".join(f"{val:{padding}g}" for val in values))
        ## END processing negative peaks
        # print terminator of current output
        print("\n" + "-"*padding*len(headers) + "\n")


def print_header(console_width:int):
    """Prints console output header

    Parameters
    ----------
    console_width : int
        console width used for string formatting, e.g. 80
    """
    # print title
    print("<" + f"> Matplotlib Peak Fit GUI ({SCRIPT_VERSION}) <".center(console_width - 2, "=") + ">")
    # print author and modifiers
    print(f"Created by {AUTHOR} @ {CREATED_AT}".rjust(console_width))
    for mod_by, mod_at in zip(MODIFIED_BY, MODIFIED_AT):
        print(f"Modified by {mod_by} @ {mod_at}".rjust(console_width))
    # output hint
    print("Output will be printed here.\n")


# <=====> main programm <=====> #
def main(file, config, console_width=80):
    print_header(console_width)
    # load data to pandas data frame
    print(f"Loading data from {file} into pandas data frame")
    df = pd.read_csv(file, **config.get("pandas_args"))
    # filter non numeric date -> handling for datetime objects needs to be implemented
    numeric_columns = df.select_dtypes(include=np.number).columns.values
    print(f"Removing {df.shape[1] - len(numeric_columns)} non numeric columns")
    df = df[numeric_columns]
    # print data frame shape and head
    print(f"Loaded data has shape: {df.shape}\n")
    print(df.head())
    # start peak fit gui
    print("\n" + "  Starting peak Fit  ".center(console_width, "-"))
    peak_fit_gui = PeakFitGUI(
        df=df,
        plot_title=f"file: {str(file)} ( rows: {df.shape[0]} x cols: {df.shape[1]} )",
        **config.get("data_defaults"),
        **config
        )
    # show figure and start interaction loop
    plt.show()
    

if __name__ == "__main__":
    # get the application directory where the application is stored
    application_directory = Path(__file__).parents[0]

    # create command line parser
    import argparse
    parser = argparse.ArgumentParser(
        prog = "python peak_fit_gui.py",
        description=(
        """
    The 'scipy.signal.find_peaks' function is a very handy utility for finding peaks in signals or other kinds of data.
    However the method is very sensible concerning its parameters. Since a manuel parameter tuning can be very frustrating I wrote 
    this little 'matplotlib' GUI application for interactively varying the 'scipy.signal.find_peaks' parameter.\n
    Currently implemented parameters:\n
                                peak height
                                peak prominence
                                peak distance
        """
        ),
        epilog=("Requirements:\n\tpython >= 3.11\n\tpackages 'scipy', 'pandas', 'matplotlib'"),
        formatter_class=argparse.RawTextHelpFormatter
    )
    # add command line arguments
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        help="Path to data file. If not supply file can be chosen via a tkinter file dialog"
        )
    parser.add_argument(
        "-c", 
        "--config", 
        type=Path, 
        default=Path(application_directory, "./settings.cfg"),
        help=f"Path to conifg file. Defaults to '{str(Path(application_directory, './settings.cfg'))}'",
        )
    # parse command line arguments
    cl_args = parser.parse_args()
    # loading config file
    with open(cl_args.config, mode="rb") as conf_file:
        config = tomllib.load(conf_file)
    # if no file is supplied via the command line interface open a tkinter file dialog
    if cl_args.file is None:
        from tkinter.filedialog import askopenfilename
        file = Path(askopenfilename())
    else:
        file = cl_args.file
    # check if file exists
    assert file.exists(), f"File {str(file)} does not exits!"
    # run main program
    main(file, config)