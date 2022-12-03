import sys
import csv

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
from matplotlib.widgets import Button, RadioButtons, Slider, RangeSlider
from scipy.signal import find_peaks

### GLOBALS ###
SCRIPT_VERSION:str = "2.0"  # current version of the script, update this if you make significant modifications

AUTHOR:str = "Moritz Kluwe"  # original script author
CREATED_AT:str = "16.11.2022"  # date of script creation

MODIFIED_BY:str = ["Moritz Kluwe"]  # list of people how contribute major modifications; change SCRIPT_VERSION variable
MODIFIED_AT:str = ["03.12.2022"]  # dates at which major modifications have taken place

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


def estimate_prominence(ydata:np.ndarray):
    mean = ydata.mean()
    return 2 * (np.abs(ydata).max() - mean)


def get_delimiter(file_path, bytes=4096):
    sniffer = csv.Sniffer()
    with open(file_path, "r") as f:
        data = f.read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter

class PeakFitGUI:
    slider_width = 0.65
    slider_height = 0.04
    slider_x0 = 0.25
    slider_y0 = 0.01
    slider_vspace = 0.015

    def __init__(
        self,
        df :pd.DataFrame,
        config:dict = None,
        plot_title:str="",
        *args,
        **kwargs
        ) -> None:
        # data frame and index definitions
        step = config.get("data_defaults", {}).get("every_nth", 1)
        df = df.iloc[::step].reset_index()
        self.df = df
        self.ncols = len(self.df.columns)
        #self.df.insert(loc=0, column="data_index", value=df.index)  # create a new column based on data index
        # set initial indexes for x and y data
        self.idx = config.get("data_defaults", {}).get("x_initial_cidx", 1)
        self.idy = config.get("data_defaults", {}).get("y_initial_cidx", 1)
        #<<<<< setup figure >>>>>
        self.fig, self.ax = plt.subplots(**config.get("figure_settings", {}))
        self.setup_figure(
            title=plot_title,
            radio_btn_settings=config.get("radio_btn_settings", {}),
            axes_settings=config.get("axes_settings", {}),
            plot_settings=config.get("plot_settings", {})
        )
        #>>>>> setup figure <<<<<
        #<<<<< read initial peak fit values >>>>>
        self.prominence = config.get("peak_fit_defaults", {}).get("prominence", (0, estimate_prominence(self.ydata)))
        self.height = config.get("peak_fit_defaults", {}).get("height", (self.ydata.min(), self.ydata.max()))
        self.width = config.get("peak_fit_defaults", {}).get("width", (0, self.xdata.size))
        self.peak_distance = config.get("peak_fit_defaults", {}).get("distance",1)
        #>>>>> read initial peak fit values <<<<<
        # dictionary to store widgets 
        self.widgets = dict()
        #<<<<< setup radio buttons >>>>>
        idx_radio_axes = self.fig.add_axes(
            [0.02, 0.46 - 0.03*self.ncols, 0.15, 0.03*self.ncols], aspect="equal"#, facecolor=(0.6,0.6,0.6)
            )
        idx_radio_axes.set_title("x data")
        idx_radio = RadioButtons(
        idx_radio_axes, 
            df.columns, active=self.idx, 
            activecolor=config.get("radio_btn_settings", {}).get("x_color", "black")
            )
        idx_radio.on_clicked(self.radio_idx_function)
        self.widgets.setdefault("radio_btn", {}).update({"idx": idx_radio})

        idy_radio_axes = self.fig.add_axes(
            [0.02, 0.6, 0.15, 0.03*self.ncols], frameon=True, aspect="equal"#, facecolor=(0.6,0.6,0.6)
            )
        idy_radio_axes.set_title("y data")
        idy_radio = RadioButtons(
            idy_radio_axes, 
            df.columns, active=self.idy, 
            activecolor =config.get("radio_btn_settings", {}).get("y_color", "black")
            )
        idy_radio.on_clicked(self.radio_idy_function)
        self.widgets.setdefault("radio_btn", {}).update({"idy": idy_radio})
        #>>>>> setup radio buttons <<<<<
        #<<<<< slider creation >>>>>
        self.width_key = self.create_slider(
            ax_dim=[self.slider_x0, self.slider_y0, self.slider_width, self.slider_height],
            label="width",
            vmin=config.get("slider_settings",{}).get("limits", {}).get("width_min", self.width[0]), 
            vmax=config.get("slider_settings",{}).get("limits", {}).get("width_max", self.width[1]),
            vinit=self.width,
            func=self.width_function,
            valstep=1,
            is_range=True,
            labelsize=config.get("slider_settings", {}).get("labelsize", 12),
            handle_style=config.get("slider_settings", {}).get("handle_style", None),
        )

        self.prominence_key = self.create_slider(
            ax_dim=[
                self.slider_x0,
                self.slider_y0 + 0.5 * (len(self.widgets["slider"])) * self.slider_height + len(self.widgets["slider"]) * self.slider_vspace,
                self.slider_width,
                self.slider_height
                ],
            label="prominence",
            vmin=config.get("slider_settings",{}).get("limits", {}).get("prom_min", 0),
            vmax=config.get("slider_settings",{}).get("limits", {}).get("prom_max",estimate_prominence(self.ydata)),
            vinit=self.prominence,
            func=self.prominence_function,
            is_range=True,
            labelsize=config.get("slider_settings", {}).get("labelsize", 12),
            handle_style=config.get("slider_settings", {}).get("handle_style", None),
        )

        self.height_key = self.create_slider(
                            ax_dim=[
                self.slider_x0,
                self.slider_y0 + 0.5 * (len(self.widgets["slider"])) * self.slider_height + len(self.widgets["slider"]) * self.slider_vspace,
                self.slider_width,
                self.slider_height
                ],
            label="height",
            vmin=self.ydata.min(),
            vmax=self.ydata.max(),
            vinit=self.height,
            func=self.height_function,
            is_range=True,
            labelsize=config.get("slider_settings", {}).get("labelsize", 12),
            handle_style=config.get("slider_settings", {}).get("handle_style", None),
        )

        self.peak_dist_key = self.create_slider(
                ax_dim=[
                self.slider_x0,
                self.slider_y0 + 0.5 * (len(self.widgets["slider"])) * self.slider_height + len(self.widgets["slider"]) * self.slider_vspace,
                self.slider_width,
                self.slider_height
                ],
            label="distance",
            vmin=config.get("slider_settings",{}).get("limits", {}).get("distance_min", 0),
            vmax=config.get("slider_settings",{}).get("limits", {}).get("distance_max", self.xdata.size),
            vinit=self.peak_distance,
            func=self.peak_distance_function,
            valstep=max(1, int(self.xdata.size / 1000)),
            labelsize=config.get("slider_settings", {}).get("labelsize", 12),
            handle_style=config.get("slider_settings", {}).get("handle_style", None),

        )
        #>>>>> slider creation <<<<<
        #<<<<< btn creation >>>>>
        axes = plt.axes([0.025, 0.14, 0.1, 0.04])
        button = Button(axes, "Print", hovercolor="0.975")
        button.on_clicked(self.print_peaks)
        self.widgets.setdefault("btn", {}).update({"print": button})

        axes = plt.axes([0.025, 0.08, 0.1, 0.04])
        button = Button(axes, "Reset", hovercolor="0.975")
        button.on_clicked(self.reset_params)
        self.widgets.setdefault("btn", {}).update({"reset": button})

        axes = plt.axes([0.025, 0.02, 0.1, 0.04])
        button = Button(axes, "Grid", hovercolor="0.975")
        button.on_clicked(self.switch_grid)
        self.widgets.setdefault("btn", {}).update({"grid": button})
        #>>>>> btn creation <<<<<
        #<<<<< hlines and vspanes >>>>>
        self.height_lines = (
            self.ax.axhline(self.height[0], **config.get("hline_settings", {})),
            self.ax.axhline(self.height[1], **config.get("hline_settings", {}))
        )

        self.height_boxes = (
            self.ax.axhspan(ymin = self.ylim[0], ymax = self.height[0], **config.get("axhspan_settings", {})),
            self.ax.axhspan(ymin = self.height[1], ymax = self.ylim[1], **config.get("axhspan_settings", {}))
        )
        #>>>>> hlines and vspanes <<<<<
        #<<<<< peak scatter >>>>>
        (self.scatter_pos_peaks,) = self.ax.plot(
            self.xdata[self.pos_peaks[0]],
            self.ydata[self.pos_peaks[0]],
            **config.get("scatter_params", {}).get("pos", {})
        )
        # scatter plot for negative peaks
        (self.scatter_neg_peaks,) = self.ax.plot(
            self.xdata[self.neg_peaks[0]],
            self.ydata[self.neg_peaks[0]],
            **config.get("scatter_params", {}).get("neg", {})
        )
        #>>>>> peak scatter <<<<<
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
            "height": self.height,
            "distance": self.peak_distance,
            "prominence": self.prominence,
            "width": self.width,
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
        param_dict["height"] = [-val for val in self.height[::-1]]
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
        ylim_lower = min(self.ydata.min(), self.height[0])
        ylim_upper = max(self.ydata.max(), self.height[1])
        delta = ylim_upper - ylim_lower
        return (
            ylim_lower - delta/10,
            ylim_upper + delta/10
        )

    def setup_figure(self, title, radio_btn_settings, axes_settings, plot_settings):
        self.fig.subplots_adjust(left=0.3, bottom=0.25)  # make space for widgets
        self.fig.suptitle("Peak Fit GUI", fontsize=16)
        self.ax.set_title(title)
        self.grid_visible = False
        self.line, *_ = self.ax.plot(
            self.xdata, 
            self.ydata, 
            **plot_settings
            ) # when 'dict.get' method is called with a key not present in the dict it returns None
        self.ax.set_xlabel(
            self.df.columns[self.idx],
            color=c if (c := radio_btn_settings.get("x_color")) else "black",
            fontsize=s if (s := axes_settings.get("x_label_size")) else 12
            )
        self.ax.set_ylabel(
            self.df.columns[self.idy],
            color=c if (c := radio_btn_settings.get("y_color")) else "black",
            fontsize=s if (s := axes_settings.get("y_label_size")) else 12
            )
        self.ax.tick_params(axis="x", labelsize=s if (s := axes_settings.get("x_tick_label_size")) else 12)
        self.ax.tick_params(axis="y", labelsize=s if (s := axes_settings.get("y_tick_label_size")) else 12)

    def create_slider(self, ax_dim:list, label:str, vmin:float, vmax:float, vinit:float, func:callable, labelsize=12, is_range=False, **kwargs) -> str:
        """
        Creates a slider and adds it to the widget dict

        :param ax_dim: dimensions of the slider axes
        :type ax_dim: list
        :param label: label of the slider which is also used as key in the widget dict
        :type label: str
        :param vmin: minimum value of the slider
        :type vmin: float
        :param vmax: maximum value of the slider
        :type vmax: float
        :param vinit: initial value of the slider
        :type vinit: float
        :param func: callback function which is called when the slider is moved
        :type func: callable
        :return: key to access the slider via the widget dict
        :rtype: str
        """
        axes = self.fig.add_axes(ax_dim)
        if is_range:
            slider = RangeSlider(
            axes,
            label=label,
            valmin=vmin,
            valmax=vmax,
            valinit=vinit,
            **kwargs,
        )
        else:
            slider = Slider(
                axes,
                label=label,
                valmin=vmin,
                valmax=vmax,
                valinit=vinit,
                **kwargs
            )
        slider.on_changed(func)
        slider.label.set_size(labelsize)
        self.widgets.setdefault("slider", {}).update({label: slider})
        return label

    def reset_slider_axes(self):
        """
        Resets the axes of all sliders based on the current slider upper and lower limits
        """
        for key, slider in self.widgets["slider"].items():
            slider.ax.set_xlim(
                slider.valmin,
                slider.valmax
            )
        return self
     
    def radio_idx_function(self, label:str):
        """
        Event function for interaction with radio_idx widget. Resets 'self.idx' to radio selection and changes axes limits and
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
        """
        Event function for interaction with radio_idy widget. Restes 'self.idy' to radio selection and changes axes limits and
        labels accordingly. Additionally resets the height slider bounds to current data range.

        Parameters
        ----------
        label : str
            string containing the selected label of the radio widget
        """
        self.idy = self.df.columns.get_loc(label)
        # reset height slider bounds
        # update line data
        self.line.set_ydata(self.ydata)
        # update y axis
        self.ax.set_ylim(self.ylim)
        self.ax.set_ylabel(label)
        self.update_peak_scatter()
        # update height slider
        self.widgets["slider"][self.height_key].valmin = self.ydata.min()
        self.widgets["slider"][self.height_key].valmax = self.ydata.max()
        self.widgets["slider"][self.height_key].set_val([self.ydata.min(), self.ydata.max()])
        # update prominence slider
        self.widgets["slider"][self.prominence_key].valmin = 0
        self.widgets["slider"][self.prominence_key].valmax = estimate_prominence(self.ydata)
        self.widgets["slider"][self.prominence_key].set_val((0, estimate_prominence(self.ydata)))
        self.reset_slider_axes()
        self.fig.canvas.draw_idle()

    def height_function(self, event:str):
        self.height = self.widgets["slider"][self.height_key].val
        set_polygon_y(
            polygon=self.height_boxes[0],
            y_lower=self.ylim[0],
            y_upper=self.height[0]
        )
        set_polygon_y(
            polygon=self.height_boxes[1],
            y_lower=self.height[1],
            y_upper=self.ylim[1]
        )
        self.update_peak_scatter()
        self.height_lines[0].set_ydata(self.height[0])
        self.height_lines[1].set_ydata(self.height[1])
        self.ax.set_ylim(self.ylim)
        plt.draw()

    def peak_distance_function(self, event):
        """Event function for peak distance slider.

        Parameters
        ----------
        event : _type_
            slider event
        """
        self.peak_distance = self.widgets["slider"][self.peak_dist_key].val
        self.update_peak_scatter()
        plt.draw()

    def prominence_function(self, event):
        self.prominence = self.widgets["slider"][self.prominence_key].val
        self.update_peak_scatter()
        plt.draw()

    def width_function(self, event):
        self.width = self.widgets["slider"][self.width_key].val
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
        for key, slider in self.widgets["slider"].items():
            slider.reset()

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
    if config.get("pandas_args")["delimiter"] == "None":
        config.get("pandas_args")["delimiter"] = get_delimiter(file)
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
        config=config
        )
    # show figure and start interaction loop
    plt.show()
    

if __name__ == "__main__":
    # get the application directory where the application is stored
    application_directory = Path(__file__).parents[0]

    # create command line parser
    import argparse
    import sys
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
        if file is None:
            sys.exit()
    else:
        file = cl_args.file
    # check if file exists
    assert file.exists(), f"File {str(file)} does not exits!"
    # run main program
    main(file, config)
