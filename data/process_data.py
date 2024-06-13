import sys
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


def clean_data(df):
    """
    Clean dataframe by splitting 'categories' into separate category columns,
    converting category values to binary (0 or 1), and removing duplicates.
    
    Args:
    df (pd.DataFrame): The dataframe to clean.
    
    Returns:
    pd.DataFrame: The cleaned dataframe.
    """
    # Check for any NaNs in the DataFrame
    if df.isnull().any().any():
        # Handle NaNs by dropping them
        df.dropna(inplace=True)

    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Use first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

    # Convert column from string to numeric
    categories = categories.apply(pd.to_numeric)

    # Validate that all values are either 0 or 1
    # Create a mask that is True for rows where all values are 0 or 1
    valid_rows = categories.apply(lambda x: x.isin([0, 1])).all(axis=1)

    # Filter the DataFrame to only include valid rows
    df = df[valid_rows]
    
    # Drop the original categories column from 'df'
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new 'categories' dataframe
    # Avoid adding duplicate columns
    df = pd.concat([df.drop(columns=categories.columns, errors='ignore'), categories], axis=1)

    # Drop duplicates more precisely based on 'id' and all category columns
    df.drop_duplicates(subset=['id'] + list(categories.columns), inplace=True)

    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to an SQLite database.
    
    Args:
    df (pd.DataFrame): The dataframe to save.
    database_filename (str): The file path of the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('CleanedData', engine, index=False, if_exists='replace')



def main():
    """
    Main function to run the ETL processes: load data, clean data, and save data.
    Utilizes command line arguments to specify file paths.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()