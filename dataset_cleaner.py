# Brandon Cossin
# Code to clean dataset
# Drops unused columns and states
# Size of CSV is dropped around 4.19X
import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')
df = df.drop(columns=['tot_cases', 'new_death', 'tot_death'], axis=1)
df = df.drop(columns=['prob_death', 'prob_cases', 'consent_cases', 'consent_deaths', 'pnew_death', 'pnew_case', 'created_at', 'conf_cases', 'conf_death'], axis=1)
df = df.loc[np.invert(df['state'].isin(['AS', 'FSM', 'PW', 'RMI', 'NYC', 'HI', 'AK', 'DC', 'GU', 'MP', 'PR', 'VI']))]
df.to_csv('cleaned_dataset.csv', index=False)
