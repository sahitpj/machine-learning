import pandas as pd

def import_data(filepath):
    data = pd.read_csv(filepath)
    return data