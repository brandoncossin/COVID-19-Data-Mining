import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[1, 2, 3, 4, 5],
    name="Positive",
    marker=dict(color="red", size=10)
))
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[5, 4, 3, 2, 1],
    name="Negative",
    marker=dict(color="blue", size=10)
))



fig.update_layout(legend_title_text='Correlation Strength')
fig.show()
