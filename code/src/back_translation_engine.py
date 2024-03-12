import pandas as pd
from modules.preprocessing import basic_processing
from modules.translate import back_translation

# Acquire the data to be translated

PATH = '../data/external/mitre-classified.xlsx'
df = pd.read_excel(PATH)

# Basic cleaning of data
df = basic_processing(df)

target = 'af' # Afrikaans
df['NameDesc_T'] = df['NameDesc'].apply(lambda x: back_translation(text=x, target=target).text)
df.to_excel(f'../data/results/translated.xlsx', index=False)
