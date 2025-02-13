# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine, inspect

def load_data(messages_filepath, categories_filepath):
    """Load and merge the data from two CSV files to a dataframe.

    INPUT:
    messages_filepath: CSV file containing the messages received by the disaster recovery teams.
    The file has 4 columns; id (message id), message (message text in English), 
    original (message in the language it was received or blank if it was originally in English),
    genre (the channel in which arrived the message).
    categories_filepath: CSV file containing the categories of the messages received.
    The file has 2 columns; id (the matching id of the message), 
    categories (a list of the 36 available categories with a 1 or a 0 at the end).

    OUTPUT:
    df: a dataframe containing the merged CVS files by id.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge both dataframes using common id
    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    """Clean the data in a dataframe

    INPUT:
    df: dataframe containing 5 columns; id (message id), message (message text in English), 
    original (message in the language it was received or blank if it was originaly in English),
    genre (the channel in which arrived the message),
    categories (a list of the 36 available categories with a 1 or a 0 at the end).

    OUTPUT:
    df: cleaned dataframe. With 40 columns; id (message id), message (message text in English), 
    original (message in the language it was received or blank if it was originaly in English),
    genre (the channel in which arrived the message),
    36 columns (one per category) with 1s and 0s.
    """
    # split the name of the categories
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    # get rid of the 0 or 1 attached to the name
    category_colnames = row.apply(lambda x: x[:-2])
    # rename columns of the dataframe
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # Drop rows where the column is not 0 or 1
        invalid_rows = categories[~categories[column].isin([0, 1])].index
        if len(invalid_rows) > 0:
            print(f"Dropping {len(invalid_rows)} rows due to invalid values in column '{column}'.")
            categories = categories.drop(invalid_rows)
            df = df.drop(invalid_rows)
    
    # Drop old 'categories' column and attach new categories (0, 1) columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Count and print how many duplicate rows exist before dropping
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Dropping {duplicate_count} duplicate rows.")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """Uploads a dataframe to a SQLite database

    INPUT:
    df: dataframe containing the data.
    database_filename: the name of the database provided by the user.
    Will be created if doesn't exists, and a new table name "Message" will be crated.
    If the database exists and contains a table called "Message" the program gives an error.

    OUTPUT:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')

    # Check if the table Message exists
    inspector = inspect(engine)
    if 'Message' in inspector.get_table_names():
        print(f"There is a table Message already in database:'{database_filename}'. It will be overwritten.")

    df.to_sql('Message', engine, if_exists='replace', index=False)


def main():
    # Make sure the user provided the required arguments
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Load the CSV data in a dataframe
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean the dataframe
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save the dataframe to a SQLite database
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