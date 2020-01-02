import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads a messages and categories CSV files and merges them into a single Pandas dataframe"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """Takes a dataframe with messages and categories and cleans it in preparation for modeling"""

    # Create a dataframe of the individual category columns
    categories = df.categories.str.split(';', expand=True)

    # Select the first row of the categories dataframe and use it to extract a list of new column
    # names for categories
    row = categories.loc[0]
    category_colnames = row.apply(lambda value: value.split('-')[0])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to binary values
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from `df`
    new_df = df.drop('categories', axis=1)

    # Concatenate the original dataframe with the new `categories` dataframe
    new_df = pd.concat([new_df, categories], axis=1)

    # Drop duplicates
    new_df.drop_duplicates(inplace=True)

    # Return the clean dataframe
    return new_df


def save_data(df, database_filename):
    """Save the dataframe into an sqlite database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('dataset', engine, index=False)


def main():
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
