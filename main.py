from loadData import load_data

if __name__ == '__main__':
    # load playlist data
    data = load_data()

    # Array of dictionary keys for indexing
    filenames = list(data.keys())
