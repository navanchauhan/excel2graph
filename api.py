from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field

from sympy import symbols, sympify

font_path = 'times_new_roman.ttf'
times_new_roman = font_manager.FontProperties(fname=font_path, style='normal')

class PlotDataRequest(BaseModel):
    x_data: list[float] = Field(..., title="X Data", description="Data Series for the X axis")
    y_data: list[float] = Field(..., title="Y Data", description="Data Series for the Y axis")
    std_dev_data: list[float] = Field([], title="Error Bars Data", description="Data Series for calculating error bars")
    label: list[str] = Field("Dataseries", title="Label", description="Label for the data series")

    color_picker: list[str] = Field(["#000000"], title="Color Picker", description="List of colors to use for the data series") 

    x_label: str = Field("X Axis", title="X Axis Label", description="Label for the X axis")
    y_label: str = Field("Y Axis", title="Y Axis Label", description="Label for the Y axis")
    title: str = Field("Plot", title="Title", description="Title of the plot")

    enable_trendline: bool = Field(True, title="Enable Trendline", description="Enable trendline")
    enable_grid: bool = Field(False, title="Enable Grid", description="Enable grid")

    x_axis_scale: str = Field("linear", title="X Axis Scale", description="Scale of the X axis")
    y_axis_scale: str = Field("linear", title="Y Axis Scale", description="Scale of the Y axis")

    trendline_equation: str = Field(None, title="Trendline Equation", description="Manually specify the equation for the trendline")
    special_mode: bool = Field(False, title="Special Mode", description="Special mode")

    constant_line_vals: list[float] = Field([], title="Constant Line Values", description="List of values for the constant line")
    constant_line_name: list[str] = Field([], title="Constant Line Names", description="List of names for the constant line")

    trendline_colour: str = Field("#000000", title="Trendline Colour", description="Colour of the trendline")
    marker_colour: str = Field("#000000", title="Marker Colour", description="Colour of the markers")

    brand_guidelines_mode: bool = Field(False, title="Brand Guidelines Mode", description="Use Brand Guidelines Mode")

app = FastAPI()

def plot_data(x_data, y_data, std_dev_data, color_picker, labels, df,
              title = "Plot", x_label = "X Axis", y_label = "Y Axis",
              plot_background_color="#ffffff", constant_line=[],
              enable_trendline=True, enable_grid=False,
              trendline_color="#000000", marker_colour="#000000",
              brand_guidelines_mode=False,
              x_axis_scale="linear", y_axis_scale="linear", trendline_equation=None, special_mode=False):
    fig, ax = plt.subplots(dpi=300)
    special_mode_color = (38/255,56/255,97/255)	

    if brand_guidelines_mode:
        font_path = "roboto-regular.ttf"
        times_new_roman = font_manager.FontProperties(fname=font_path, style='normal')
    else:
        font_path = 'times_new_roman.ttf'
        times_new_roman = font_manager.FontProperties(fname=font_path, style='normal')

    plots = []

    for idx, _ in enumerate(x_data):
        x = df[x_data[idx]].astype(float)
        y = df[y_data[idx]].astype(float)
        #color = color_picker[idx]
        data_series_title = labels[idx]
        if special_mode:
            if (std_dev_data[idx] != None):
                plot = ax.errorbar(x, y, yerr=df[std_dev_data[idx]].astype(float), fmt='o', ecolor=special_mode_color, capsize=5, color=special_mode_color, label=data_series_title)
            else:
                plot = ax.plot(x, y, 'o', color=special_mode_color, label=data_series_title)
        else:
            if (std_dev_data[idx] != None):
                plot = ax.errorbar(x, y, yerr=df[std_dev_data[idx]].astype(float), fmt='o', ecolor='black', capsize=5, color=marker_colour, label=data_series_title)
            else:
                plot = ax.plot(x, y, color=marker_colour, label=data_series_title)

        if (type(plot) == list):
            plots.extend(plot)
        else:
            plots.append(plot)

    handles = plots

    if trendline_equation == "":
        trendline_equation = None

    if enable_trendline:
        if special_mode:
            trendline_color = special_mode_color
        if trendline_equation != None:
            try:
                x = symbols('x')
                p = sympify(trendline_equation)
                x_range = np.linspace(df[x_data[0]].astype(float).min(), df[x_data[0]].astype(float).max(), 100)
                y_range = [p.subs(x, i) for i in x_range]
                h, = ax.plot(x_range, y_range, linestyle="dashdot", label="Trendline", color=trendline_color)
                handles.append(h)
            except:
                print("Invalid Equation")
        else:   
            x = df[x_data[0]].astype(float)
            y = df[y_data[0]].astype(float)
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            h, = ax.plot(x,p(x), linestyle="dashdot", label="Trendline", color=trendline_color)
            handles.append(h)

    if not brand_guidelines_mode:
        light_grey = 0.9
        dar_grey = 0.4

        for idx, line in enumerate(constant_line):
            val, name = line
            idx += 1
            grey_shade = light_grey - (light_grey - dar_grey) * (idx / len(constant_line))
            color = (grey_shade, grey_shade, grey_shade)
            h = ax.axhline(y=val, linestyle='dashed', dashes=(idx+3,2)  , color=color, label=name)
            handles.append(h)
    else:
        light_green = (139/255, 196/255, 63/255)
        dark_green = (21/255, 130/255, 56/255)

        for idx, line in enumerate(constant_line):
            val, name = line
            idx += 1
            ratio = idx / len(constant_line)
            color = (
                light_green[0] + (dark_green[0] - light_green[0]) * ratio,
                light_green[1] + (dark_green[1] - light_green[1]) * ratio,
                light_green[2] + (dark_green[2] - light_green[2]) * ratio
                )
            h = ax.axhline(y=val, linestyle='dashed', dashes=(idx+3,2)  , color=color, label=name)
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

    if not special_mode:
        fig.patch.set_facecolor(plot_background_color)
        fig.tight_layout(pad=3.0)
    #ax.invert_xaxis()
    

    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)

    if special_mode:
        # Hide title, x label, y label
        ax.legend().set_visible(False)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)
        arial_font = FontProperties(fname='./arial.ttf')
        mpl.font_manager.fontManager.addfont('./arial.ttf')
        with mpl.rc_context({"font.family": arial_font.get_name(), "font.size": 9}):
            ax.legend(handles, labels, loc='best')

            # set axis tick labels to Arial:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(arial_font)

            return fig

    return fig

@app.post("/post")
async def create_plot(request: PlotDataRequest, data: Request):

    print(data)

    if len(request.x_data) != len(request.y_data):
        raise HTTPException(status_code=400, detail="X and Y data must be the same length")

    if len(request.constant_line_vals) != len(request.constant_line_name):
        raise HTTPException(status_code=400, detail="Constant line values and names must be the same length")

    constant_line = list(zip(request.constant_line_vals, request.constant_line_name))

    # Create DF from request.x_data and request.y_data -> assign column header as x_data and y_data
    df = pd.DataFrame(list(zip(request.x_data, request.y_data)), columns=["X", "Y"])

    # if request.std_dev_data exists, add to df
    if len(request.std_dev_data) > 0:
        df["STD_DEV"] = request.std_dev_data

    x_data = ["X"]
    y_data = ["Y"]

    if len(request.std_dev_data) > 0:
        std_dev_data = ["STD_DEV"]
    else:
        std_dev_data = [None]

    labels = request.label

    print(f"Got trendline colour {request.trendline_colour}")

    fig = plot_data(x_data, y_data, std_dev_data, request.color_picker, labels, df,
                    title=request.title, x_label=request.x_label, y_label=request.y_label,
                    enable_trendline=request.enable_trendline, enable_grid=request.enable_grid,
                    constant_line=constant_line, trendline_color=request.trendline_colour,
                    marker_colour=request.marker_colour, brand_guidelines_mode=request.brand_guidelines_mode,
                    x_axis_scale=request.x_axis_scale, y_axis_scale=request.y_axis_scale, trendline_equation=request.trendline_equation,
                    special_mode=request.special_mode
                    )

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
    





