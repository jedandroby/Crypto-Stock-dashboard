import numpy as np
import panel as pn
from pathlib import Path
import pandas as pd
import hvplot.pandas

# Read the the ny_places_interest.csv file into a DataFrame
places_of_interest = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/geo-deploy/Resources/nyc_excursion_plans.csv'
).dropna()

arrival_and_first_location = places_of_interest[
    (places_of_interest["Name"].str.contains("Airport"))
    | (places_of_interest["Name"].isin(["Aqueduct Race Track"]))
]
# Plot the arriving airport and the first location
# Set size = 300 to make the points easier to see on the map
first_route = arrival_and_first_location.hvplot.points(
    'Longitude', 
    'Latitude', 
    geo=True, 
    color='Name',
    size = 300,
    tiles='OSM',
    frame_width = 700,
    frame_height = 500
    )

first_second_third_locations = places_of_interest[
    places_of_interest["Name"].isin(
        ["Aqueduct Race Track", "Juniper Valley Park", "Madison Square"]
    )
]

# Create the plot including your first, second and third locations
# Set size = 300 to make the points easier to see on the map
second_route = first_second_third_locations.hvplot.points(
    'Longitude', 
    'Latitude', 
    geo=True, 
    color='Name',
    size = 300,
    tiles='OSM',
    frame_width = 700,
    frame_height = 500
    )

## Step 5: Plot the route between your third, fourth, and fifth locations.
third_fourth_fifth_locations = places_of_interest[
    places_of_interest["Name"].isin(
        ["Madison Square", "Liberty Island", "Ellis Island"]
    )
]
# Create the plot including your third, fourth and fifth locations
# Set size = 300 to make the points easier to see on the map
third_route = third_fourth_fifth_locations.hvplot.points(
    'Longitude', 
    'Latitude', 
    geo=True, 
    color='Name',
    alpha=0.8,
    size = 300,
    tiles='OSM',
    frame_width = 700,
    frame_height = 500
    )

# Create a Title for the Dashboard
title = pn.pane.Markdown(
    """
# Real Estate Analysis of San Francisco from 2010 to 2016
""",
    width=800,
)

welcome = pn.pane.Markdown(
    """
This dashboard presents a visual analysis of ...
"""
)
# Create a tab layout for the dashboard
tabs = pn.Tabs(
    ("Welcome", pn.Column(welcome, first_route)),
    ("First Route", pn.Row(first_route)),
    ("Second Route", pn.Column(second_route)),
    ("Third Route", pn.Column(third_route,width=960))
    )

pn.Column(pn.Row(title), tabs, width=900).servable(target='main')
# pn.extension(template='fast')

# freq = pn.widgets.FloatSlider(
#     name='Frequency', start=0, end=10, value=5
# ).servable(target='sidebar')

# ampl = pn.widgets.FloatSlider(
#     name='Amplitude', start=0, end=1, value=0.5
# ).servable(target='sidebar')

# def plot(freq, ampl):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     xs = np.linspace(0, 1)
#     ys = np.sin(xs*freq)*ampl
#     ax.plot(xs, ys)
#     return fig

# pn.Column(
#     '# Sine curve',
#     pn.bind(plot, freq, ampl),
# ).servable(target='main')