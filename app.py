from flask import Flask, render_template, request, redirect, url_for, flash
import json

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import re

font_path = 'times_new_roman.ttf'
times_new_roman = font_manager.FontProperties(fname=font_path, style='normal')

"""
    fig = plot_data(x_data, y_data, std_dev_data, color_picker, title=plot_title,
                    x_label=x_axis_label, y_label=y_axis_label,
                    plot_background_color=plot_background_color,
                    constant_line=constant_lines,
                    enable_trendline=enable_trendline,
                    trendline_color=color_picker_trendline)
"""

def plot_data(x_data, y_data, std_dev_data, color_picker, labels, df,
              title = "Plot", x_label = "X Axis", y_label = "Y Axis",
              plot_background_color="#ffffff", constant_line=[],
              enable_trendline=True, enable_grid=False,
              trendline_color="#000000", x_axis_scale="linear", y_axis_scale="linear"):
    fig, ax = plt.subplots(dpi=300)

    plots = []

    for idx, _ in enumerate(x_data):
        x = df[x_data[idx]].astype(float)
        y = df[y_data[idx]].astype(float)
        color = color_picker[idx]
        data_series_title = labels[idx]
        #print(df[x][0], df[y][0], color, data_series_title)
        if (std_dev_data[idx] != None):
            plot = ax.errorbar(x, y, yerr=df[std_dev_data[idx]].astype(float), fmt='o', ecolor='black', capsize=5, color=color, label=data_series_title)
        else:
            plot = ax.plot(x, y, 'o', color=color, label=data_series_title)

        if (type(plot) == list):
            plots.extend(plot)
        else:
            plots.append(plot)

    handles = plots

    if enable_trendline:
        x = df[x_data[0]].astype(float)
        y = df[y_data[0]].astype(float)
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        h, = ax.plot(x,p(x), linestyle="dashed", label="Trendline", color=trendline_color)
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
    

    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)

    return fig


app = Flask(__name__)

def create_df_from_textarea(textarea):
    rows = textarea.split('\n')
    data = [row.split("\t") for row in rows]
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

@app.route('/')
def index():
    return render_template('index.html')

# POST for process_data
@app.route('/process_data', methods=['POST'])
def process_data():
    print(request.form)

    textarea = request.form['excelData']
    df = create_df_from_textarea(textarea)

    print(df)

    constant_lines = []
    for x in range(0,10):
        try:
            y = float(request.form['constantLine' + str(x)])
            name = request.form['constantLineLabel' + str(x)]
            constant_lines.append((y, name))
        except KeyError:
            break

    data_keys = f"{request.form.keys()}"
    print(data_keys)

    xPattern = r'xColumn-\d+'
    yPattern = r'yColumn-\d+'
    stdDevPattern = r'stdDevColumn-\d+'
    colorPickerPattern = r'colorPicker-\d+'
    labelPattern = r'dataSeries-\d+'


    # match in data_keys string
    x_data_matches = re.findall(xPattern, data_keys)
    y_data_matches = re.findall(yPattern, data_keys)
    std_dev_data_matches = re.findall(stdDevPattern, data_keys)
    color_picker_matches = re.findall(colorPickerPattern, data_keys)
    label_matches = re.findall(labelPattern, data_keys)

    print(x_data_matches, y_data_matches, std_dev_data_matches, color_picker_matches)

    # Not sure if we need to sort these
    #x_data_matches.sort()
    #y_data_matches.sort()
    #std_dev_data_matches.sort()

    x_data = []

    for x in x_data_matches:
        val = request.form.get(x)
        if val != '':
            x_data.append(df.columns[int(val)])
        else:
            x_data.append(None)

    y_data = []

    for y in y_data_matches:
        val = request.form.get(y)
        if val != '':
            y_data.append(df.columns[int(val)])
        else:
            y_data.append(None)

    std_dev_data = []

    for std_dev in std_dev_data_matches:
        val = request.form.get(std_dev)
        if val != '':
            std_dev_data.append(df.columns[int(val)])
        else:
            std_dev_data.append(None)

    color_picker = []

    for color in color_picker_matches:
        val = request.form.get(color)
        if val != '':
            color_picker.append(val)
        else:
            color_picker.append(None)

    data_series_label = []

    for label in label_matches:
        val = request.form.get(label)
        if val != '':
            data_series_label.append(val)
        else:
            data_series_label.append("Data")


    print(x_data)
    print(y_data)
    print(std_dev_data)
    print(color_picker)
    print(data_series_label)

    

    x_axis_label = request.form.get('xTitle', 'X Axis')
    y_axis_label = request.form.get('yTitle', 'Y Axis')
    plot_title = request.form.get('plotTitle', 'Plot Title')

    plot_background_color = request.form.get('colorPickerPlotBackground', '#ffffff')
    color_picker_trendline = request.form.get('colorPickerTrendline', '#00ff00')

    x_axis_scale = request.form.get('xAxisScale', 'linear')
    y_axis_scale = request.form.get('yAxisScale', 'linear')

    calc_trendline = request.form.get('calcTrendline', 'off')
    if calc_trendline == 'on':
        enable_trendline = True
    else:
        enable_trendline = False

    fig = plot_data(x_data, y_data, std_dev_data, color_picker, data_series_label, df, title=plot_title,
                    x_label=x_axis_label, y_label=y_axis_label,
                    plot_background_color=plot_background_color,
                    constant_line=constant_lines,
                    enable_trendline=enable_trendline,
                    trendline_color=color_picker_trendline,
                    x_axis_scale=x_axis_scale,
                    y_axis_scale=y_axis_scale)

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
