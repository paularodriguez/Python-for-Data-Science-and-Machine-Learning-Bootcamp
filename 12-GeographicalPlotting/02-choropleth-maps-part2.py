import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# region

df = pd.read_csv('2014_World_GDP')

print(df.head())

#           COUNTRY  GDP (BILLIONS) CODE
# 0     Afghanistan           21.71  AFG
# 1         Albania           13.40  ALB
# 2         Algeria          227.80  DZA
# 3  American Samoa            0.75  ASM
# 4         Andorra            4.80  AND

data = dict(type='choropleth',
            locations=df['CODE'],
            z=df['GDP (BILLIONS)'],
            text=df['COUNTRY'],
            colorbar={'title': 'GDP in Billions USD'})

layout = dict(title='2014 global GDP',
              geo=dict(showframe=False,
                       projection={'type': 'mercator'}))

choromap = go.Figure(data=[data], layout=layout)
plot(choromap, filename='02.map-world-gdp.html')

# endregion
