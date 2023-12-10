from flask import Flask, render_template, request, redirect, url_for, flash
import json

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

font_path = 'times_new_roman.ttf'
times_new_roman = font_manager.FontProperties(fname=font_path, style='normal')

def plot_data(
        x_data, y_data, std_dev_data=None,
        title=None, x_label=None, y_label=None,
        data_point_color='#ff0000', plot_background_color='#ffffff', constant_line=[],
        enable_trendline=True, enable_grid=False,
        data_series_title="Data Series",
        trendline_color='#00ff00'
        ):
    fig, ax = plt.subplots(dpi=300)

    if (std_dev_data):
        main_plot = ax.errorbar(x_data, y_data, yerr=std_dev_data, fmt='o', ecolor='black', capsize=5,  label=data_series_title, color=data_point_color)
    else:
        main_plot = ax.plot(x_data, y_data, 'o', color=data_point_color, label=data_series_title)

    handles = [main_plot]

    if enable_trendline:
        z = np.polyfit(x_data, y_data, 2)
        p = np.poly1d(z)
        h, = ax.plot(x_data,p(x_data), linestyle="dashed", label="Trendline", color=trendline_color)
        handles.append(h)

    light_grey = 0.9
    dar_grey = 0.4

    for idx, line in enumerate(constant_line):
        val, name = line
        idx += 1
        grey_shade = light_grey - (light_grey - dar_grey) * (idx / len(constant_line))
        color = (grey_shade, grey_shade, grey_shade)
        h = ax.axhline(y=val, linestyle='--', color=color, label=name)
        handles.append(h)

    ax.grid(True,linestyle=(0,(1,5))) # enable_grid)

    ax.set_facecolor(plot_background_color)
    ax.set_xlabel(x_label, fontproperties=times_new_roman)
    ax.set_ylabel(y_label, fontproperties=times_new_roman)
    title = title.replace(' in ', '\nin ')
    ax.set_title(title, wrap=True, fontproperties=times_new_roman)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(times_new_roman)

    print(handles)
    print(handles[0])
    print(handles[0].get_label())
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='best', prop=times_new_roman)

    fig.patch.set_facecolor(plot_background_color)
    fig.tight_layout(pad=3.0)
    #ax.invert_xaxis()
    #ax.set_xscale("log")

    return fig


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# POST for process_data
@app.route('/process_data', methods=['POST'])
def process_data():
    print(request.form)
    x_data_json = request.form.get('xData', '[]')
    y_data_json = request.form.get('yData', '[]')
    std_dev_data_json = request.form.get('stdDevData', '[]')

    constant_lines = []
    for x in range(0,10):
        try:
            y = float(request.form['constantLine' + str(x)])
            name = request.form['constantLineLabel' + str(x)]
            constant_lines.append((y, name))
        except KeyError:
            break

    x_data = json.loads(x_data_json)
    print(x_data, x_data_json)
    y_data = json.loads(y_data_json)
    std_dev_data = json.loads(std_dev_data_json)

    # Process data
    x_data = [float(x) for x in x_data]
    y_data = [float(y) for y in y_data]
    std_dev_data = [float(std_dev) for std_dev in std_dev_data] if std_dev_data else []

    x_axis_label = request.form.get('xTitle', 'X Axis')
    y_axis_label = request.form.get('yTitle', 'Y Axis')
    plot_title = request.form.get('plotTitle', 'Plot Title')

    data_point_color = request.form.get('colorPickerDP', '#ff0000')
    plot_background_color = request.form.get('colorPickerPlotBackground', '#ffffff')
    color_picker_trendline = request.form.get('colorPickerTrendline', '#00ff00')

    data_series_title = request.form.get('dataSeriesTitle', 'Data Series')

    calc_trendline = request.form.get('calcTrendline', 'off')
    if calc_trendline == 'on':
        enable_trendline = True
    else:
        enable_trendline = False

    fig = plot_data(
        x_data, y_data, std_dev_data=std_dev_data,
        title=plot_title, x_label=x_axis_label, y_label=y_axis_label,
        data_point_color=data_point_color, plot_background_color=plot_background_color,
        constant_line=constant_lines, data_series_title=data_series_title,
        enable_trendline=enable_trendline,
        trendline_color=color_picker_trendline
    )

    # Return plot as image
    from io import BytesIO
    import base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')

    # Embed base64 image in HTML
    return '<img src="data:image/png;base64,{}">'.format(image_base64)

if __name__ == '__main__':
    app.run(port=8080,debug=True)