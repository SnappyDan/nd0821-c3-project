import pandas as pd

DATA_PATH = '../data/census.csv'
CORRECTED_DATA_PATH = '../data/census_cor.csv'

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace(' ', '')
df.to_csv(CORRECTED_DATA_PATH)
