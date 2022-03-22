import pandas as pd

DATA_PATH = '../data/census.csv'
CORRECTED_DATA_PATH = '../data/census_cor.csv'

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace(' ', '')
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

for col in cat_features:
    df[col] = df[col].str.replace(' ', '')

df.to_csv(CORRECTED_DATA_PATH, index=False)


