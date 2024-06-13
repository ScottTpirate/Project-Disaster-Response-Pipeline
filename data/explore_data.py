import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    
    Args:
    messages_filepath (str): The file path for the messages dataset.
    categories_filepath (str): The file path for the categories dataset.
    
    Returns:
    pd.DataFrame: A dataframe merging messages and categories on 'id'.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


df = load_data("data/disaster_messages.csv", "data/disaster_categories.csv")

for col in df:
    print(col)
    print(df[col].value_counts())