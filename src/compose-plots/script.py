import numpy as np
import panel as pn
from pathlib import Path
import pandas as pd
import hvplot.pandas



hospital_data = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/hospital_claims.csv'
).dropna()

# Slice the DataFrame to consist of only "552 - MEDICAL BACK PROBLEMS W/O MCC" information
procedure_552_charges = hospital_data[
    hospital_data["DRG Definition"] == "552 - MEDICAL BACK PROBLEMS W/O MCC"
]
# Group data by state and average total payments, and then sum the values
payments_by_state = procedure_552_charges[["Average Total Payments", "Provider State"]]
# Sum the average total payments by state
total_payments_by_state = payments_by_state.groupby("Provider State").sum()
plot1 = total_payments_by_state.hvplot.bar(rot = 45)


# Sort the state data values by Average Total Paymnts
sorted_total_payments_by_state = total_payments_by_state.sort_values("Average Total Payments")
sorted_total_payments_by_state.index.names = ['Provider State Sorted']
# Plot the sorted data
plot2 = sorted_total_payments_by_state.hvplot.line(rot = 45)

sorted_total_payments_by_state.index.names = ['Provider State Sorted']
plot3 = total_payments_by_state.hvplot.bar(rot = 45) + sorted_total_payments_by_state.hvplot(rot = 45)

# Group data by state and average medicare payments, and then sum the values
medicare_payment_by_state = procedure_552_charges[["Average Medicare Payments", "Provider State"]]
total_medicare_by_state = medicare_payment_by_state.groupby("Provider State").sum()
# Sort data values
sorted_total_medicare_by_state = total_medicare_by_state.sort_values("Average Medicare Payments")
plot4 = sorted_total_medicare_by_state.hvplot.bar(rot = 45)

plot5 = sorted_total_payments_by_state.hvplot.line(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", rot = 45)

# Overlay plots of the same type using * operator
plot6 = sorted_total_payments_by_state.hvplot.bar(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", width = 1000, rot = 45)



pn.extension( template="fast")

pn.state.template.param.update(
    # site_url="",
    # site="",
    title="UCBerkely FinTech Bootcamp Demo",
    favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
# Create a Title for the Dashboard
title = pn.pane.Markdown(
    """
# Hospital Data Analysis - UCBerkley FinTech Bootcamp Demo
""",
    width=800,
)
image = pn.pane.image.PNG(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    alt_text='Meme Logo',
    link_url='https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    width=500
)
welcome = pn.pane.Markdown(
    """
### This dashboard presents a visual analysis of hospital data for a demo to UCBerkley FinTech Bootcamp students in `Firas Obeid's` classes
* Motive is to keep students up to date with the tools that allows them to define a problem till deployment in a very short amount of time for efficient deliverables in the work place or in academia.
* Disclaimer: All data presented are from UCBerkley resources.
* Disclaimer: All references: https://blog.holoviz.org/panel_0.14.html

"""
)
# Create a tab layout for the dashboard
tabs = pn.Tabs(
    ("Welcome", pn.Column(welcome, image)),
    ("total_payments_by_state", pn.Row(plot1)),
    ("sorted_total_payments_by_state", pn.Row(plot2)),
    ("Tab1 + Tab2", pn.Column(plot3,width=960)),
    ("sorted_total_medicare_by_state", pn.Row(plot4,plot5, plot6, width=2000))
    )

pn.Column(pn.Row(title), tabs, width=900).servable(target='main')