import pandas as pd
import os

def load_data():
    # Load the data for each of our csv files
    # This is just to save time and not have to call the Spotify API every time

    # For each .csv file in the directory, load the .csv file back into a pandas dataframe
    data = {}
    filenames = []
    for file in os.listdir("./data"):  
        if file.endswith(".csv"):
            # get the name and read the csv
            filenames.append(file)
            df = pd.read_csv("./data/" + file)

            # Add the dataframe to the dictionary
            data[file] = df

    return data, filenames