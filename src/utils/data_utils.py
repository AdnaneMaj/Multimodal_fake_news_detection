import pandas as pd
from ..Enums import BaseEnum

def get_dataframe():
    # Get the csv file of the data as a dataframe
    df = pd.read_csv(BaseEnum.CSV_DATA_PATH.value)
    return df