from loadData import load_data
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # get list of audio characteristics
    features_full = ["danceability","energy","loudness", 
                     "speechiness","instrumentalness",
                     "liveness","valence","tempo", "duration_ms","popularity"]

    features = ["danceability","energy","loudness", "liveness","valence","tempo", "duration_ms","popularity"]

    # load playlist data
    data = load_data()

    # Array of playlist names for indexing and actual playlists
    playlists = data[0]
    names = data[1]

    # Create a hash of playlist tuples
    pairs = [(playlists[names[0]], playlists[names[6]]), 
            (playlists[names[1]], playlists[names[7]]),
            (playlists[names[2]], playlists[names[8]]),
            (playlists[names[3]], playlists[names[9]]),
            (playlists[names[4]], playlists[names[10]]),
            (playlists[names[5]], playlists[names[11]])]
    pair_names = names[0:6]

    # dataframes to hold chi-squared and t-test results
    chi_results = pd.DataFrame(index=np.arange(len(features)), columns=pair_names)
    t_results = pd.DataFrame(index=np.arange(len(features)), columns=pair_names)

    # loop over each pair in pairs
    for index, pair in enumerate(pairs):
        # get the two playlists
        playlist1 = pair[0]
        playlist2 = pair[1]

        # loop over the features and plot histograms for each feature
        fig, axs = plt.subplots(2, 4, figsize=(30, 20))
        for feature in features:
            plt.subplot(2, 4, features.index(feature) + 1)
            plt.hist(playlist1[feature], bins=20, alpha=0.5, label=playlist1["playlist_name"][0])
            plt.hist(playlist2[feature], bins=20, alpha=0.5, label=playlist2["playlist_name"][0])
            plt.legend()
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # perform a chi-squared test on each histogram
            chi_squared, p_value = stats.chisquare(playlist1[feature])
            chi_squared_2, p_value_2 = stats.chisquare(playlist2[feature])
            chi_results.iloc[features.index(feature), index] = (p_value, p_value_2)

            # perform a t-test on each histogram
            t_statistic, p_value = stats.ttest_ind(playlist1[feature], playlist2[feature])
            t_results.iloc[features.index(feature), index] = p_value

        # save each plot to a jpeg and put them in a /images/ folder
        fig.suptitle("Feature Histograms for " + playlist1["playlist_name"][0] + " and " + playlist2["playlist_name"][0])
        plt.savefig('./images/' + playlist1["playlist_name"][0] + '_histogram.jpeg')

        # loop over the features and plot a probability plot for playlist 1
        fig, axs = plt.subplots(2, 4, figsize=(30, 20))
        for feature in features:
            # plot a histogram of each feature in the subplot
            plt.subplot(2, 4, features.index(feature) + 1)
            stats.probplot(playlist1[feature], dist="norm", plot=plt)
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
        # save each plot to a jpeg and put them in a /images/ folder
        fig.suptitle("Feature Probability Plots for " + playlist1["playlist_name"][0])
        plt.savefig('./images/' + playlist1["playlist_name"][0] + '_probplot.jpeg')

        # loop over the features and plot a probability plot for playlist 2
        fig, axs = plt.subplots(2, 4, figsize=(30, 20))
        for feature in features:
            # plot a histogram of each feature in the subplot
            plt.subplot(2, 4, features.index(feature) + 1)
            stats.probplot(playlist2[feature], dist="norm", plot=plt)
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
        # save each plot to a jpeg and put them in a /images/ folder
        fig.suptitle("Feature Probability Plots for " + playlist2["playlist_name"][0])
        plt.savefig('./images/' + playlist2["playlist_name"][0] + '_probplot.jpeg')

    # save the chi-squared and t-test results to a csv
    chi_results.to_csv('./results/chi_results.csv')
    t_results.to_csv('./results/t_results.csv')


        
