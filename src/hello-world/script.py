import panel as pn
pn.extension(sizing_mode="stretch_width", template="fast")
pn.state.template.param.update(
    site="Panel Dashboards",
    title="Hello World",
    favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
pn.panel(
    "Hello and welcome to the awesome world of [Panel](https://panel.holoviz-org) and data apps"
).servable()