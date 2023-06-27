default_config = """
# arguments passed to pd.read_csv
# see https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
[pandas_args]
#delimiter = ","  # comment this line for autodetection

# initial columns selected for x and y data use 0 for index column
[data_defaults]
x_initial_cidx = 1
y_initial_cidx = 2
every_nth = 1  # value used for slicing data [::every_nth]

# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
[peak_fit_defaults]
# height = [-0.3, 0]  # [lower limit, upper limit], comment line to use default min, max data range
# prominence = [0, 1] # [min, max], comment line to use default [0, 1]
#distance = 1  # comment line to use default 1


# colors for radio active button
[radio_btn_settings]
x_color = [0.7, 0.0, 0.0]
y_color = [0.0, 0.0, 0.7]
x_font_size = 14
y_font_size = 14

[slider_settings]
labelsize = 14
[slider_settings.handle_style]
facecolor = [0.8, 0.8, 0.8]
edgecolor = "gray"
size = 14
[slider_settings.limits]
#width_min = 0
#width_max = 0
#prom_min = 0
#prom_max = 0
#distance_min = 0
#distance_max = 0

# paramter passed to matplotlib.pyplot.plot for scattering positve peaks
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
[scatter_params]
[scatter_params.pos]
color = "red"
marker = "o"
markersize = 7
markerfacecolor = "None"
linestyle = "None"
label = "positive peaks"
zorder = 10
# paramter passed to matplotlib.pyplot.plot for scattering negative peaks
[scatter_params.neg]
color = "blue"
marker = "s"
markersize = 7
markerfacecolor = "None"
linestyle = "None"
label = "negative peaks"
zorder = 10

# paramter passed to matplotlib.pyplot.figure for configuring main figure
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
[figure_settings]
figsize = [16, 9]
facecolor=[0.9, 0.9, 0.9]

[axes_settings]
x_label_size = 22
x_tick_label_size = 14
y_label_size = 22
y_tick_label_size = 14

# paramter passed to matplotlib.pyplot.plot for plotting csv data
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
[plot_settings]
color = "black"
linestyle = "-"
linewidth = "1"

# paramter passed to matplotlib.pyplot.axhlin for displaying height bounds
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhline.html
[hline_settings]
color = "red"
linestyle = "--"
alpha = 0.5

# parameters passed to matplotlib.pyplot.axhspan for displaying "out of range" box
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhspan.html
[axhspan_settings]
color = [0.6, 0.4, 0.4]
alpha = 0.2
"""