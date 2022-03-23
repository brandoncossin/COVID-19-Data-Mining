#generic code to generate a used legend for the cytoscape
import plotly.graph_objects as go
import numpy as np


t = np.linspace(0, 10, 100)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=np.sin(t),
    name='Increasing',
    mode='markers',
    marker_color='#FFA500'
))

fig.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='Decreasing',
    marker_color='green'
))
fig.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='No Change',
    marker_color='grey'
))
# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='Styled Scatter',
                  yaxis_zeroline=False, xaxis_zeroline=False)



fig.update_layout(legend_title_text='New Case Change')
fig.show()
