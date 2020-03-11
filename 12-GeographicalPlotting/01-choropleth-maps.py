import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# init_notebook_mode(connected=True)

# region USA map
data = dict(type='choropleth',
            # States codes
            locations=['AZ', 'CA', 'NY'],
            locationmode='USA-states',
            colorscale='Portland',
            # Tooltip for each location
            text=['Arizona', 'California', 'New York'],
            # Value for each location
            z=[1.0, 2.0, 3.0],
            colorbar={'title': 'Colorbar Title Goes Here'})

layout = dict(geo={'scope': 'usa'})
choromap = go.Figure(data=[data], layout=layout)

plot(choromap, filename='01.map-usa.html')
# endregion

# region exports
df = pd.read_csv('2011_US_AGRI_Exports')
print(df.head())

#   code        state  ...   cotton                                               text
# 0   AL      Alabama  ...   317.61  Alabama<br>Beef 34.4 Dairy 4.06<br>Fruits 25.1...
# 1   AK       Alaska  ...     0.00  Alaska<br>Beef 0.2 Dairy 0.19<br>Fruits 0.0 Ve...
# 2   AZ      Arizona  ...   423.95  Arizona<br>Beef 71.3 Dairy 105.48<br>Fruits 60...
# 3   AR     Arkansas  ...   665.44  Arkansas<br>Beef 53.2 Dairy 3.53<br>Fruits 6.8...
# 4   CA   California  ...  1064.95   California<br>Beef 228.7 Dairy 929.95<br>Frui...

data_exports = dict(type='choropleth',
                    colorscale='YlOrRd',
                    locations=df['code'],
                    locationmode='USA-states',
                    z=df['total exports'],
                    text=df['text'],
                    marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                    colorbar={'title': 'Millions USD'})

layout_exports = dict(title='2011 US Agriculture Exports by State',
                      geo=dict(scope='usa', showlakes=True, lakecolor='rgb(85,173,240)'))
choromap_exports = go.Figure(data=[data_exports], layout=layout_exports)

plot(choromap_exports, filename='01.map-usa-exports.html')

# endregion
