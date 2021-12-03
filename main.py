import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from loadData import load_data


# The first analysis is on the emotional characteristics
def AnalysisOne():
    features = ["danceability","energy","loudness","liveness","valence","tempo","duration_s","popularity"]

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
    stat_results = pd.DataFrame(index=np.arange(len(features)), columns=pair_names)

    # adjust font size for all plt plots
    matplotlib.rcParams.update({'font.size': 30})

    # loop over each pair in pairs
    for index, pair in enumerate(pairs):
        # get the two playlists
        playlist1 = pair[0]
        playlist2 = pair[1]

        # loop over the features and plot histograms for each feature
        fig, axs = plt.subplots(2, 4, figsize=(40, 30))
        for feature in features:
            plt.subplot(2, 4, features.index(feature) + 1)
            plt.hist(playlist1[feature], bins=20, alpha=0.5, label=playlist1["playlist_name"][0])
            plt.hist(playlist2[feature], bins=20, alpha=0.5, label=playlist2["playlist_name"][0])
            plt.legend()
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # place mean, mode, and standard deviation for both playlists in a dataframe
            mean = (np.mean(playlist1[feature]), np.mean(playlist2[feature]))
            mode = (stats.mode(playlist1[feature]), stats.mode(playlist2[feature]))
            std = (np.std(playlist1[feature]), np.std(playlist2[feature]))    

            # place mean, mode, std into a single array and then into the dataframe
            stats_array = np.array([mean, mode, std], dtype=object)
            stat_results.iloc[features.index(feature), index] = stats_array  

            # perform a chi-squared test on each histogram
            chi_squared, p_value = stats.normaltest(playlist1[feature])
            chi_squared_2, p_value_2 = stats.normaltest(playlist2[feature])

            # turn the p value(s) into a 1 if it is greater than 0.05
            p_value = 1 if p_value > 0.05 else 0
            p_value_2 = 1 if p_value_2 > 0.05 else 0
            chi_results.iloc[features.index(feature), index] = (p_value, p_value_2)

            # perform a t-test on each histogram if the chi-square result is normal 
            if p_value == 1 and p_value_2 == 1:
                t_statistic, p_value = stats.ttest_ind(playlist1[feature], playlist2[feature])
                p_value = 1 if p_value > 0.05 else 0
                t_results.iloc[features.index(feature), index] = p_value
            else:
                t_results.iloc[features.index(feature), index] = str(p_value) + "(NN)"

        # save each plot to a jpeg and put them in a /images/ folder
        fig.suptitle("Feature Histograms for " + playlist1["playlist_name"][0] + " and " + playlist2["playlist_name"][0])
        plt.savefig('./images/' + playlist1["playlist_name"][0] + '_histogram.jpeg')

        # loop over the features and plot a probability plot for playlist 1
        fig, axs = plt.subplots(2, 4, figsize=(40, 30))
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
        fig, axs = plt.subplots(2, 4, figsize=(40, 30))
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

    # save the chi-squared and t-test results, and statistics to a csv
    chi_results.to_csv('./results/chi_results.csv')
    t_results.to_csv('./results/t_results.csv')
    stat_results.to_csv('./results/stat_results.csv')

# The second analysis will look at each API feature in more detail and produce a linear regression
def AnalysisTwo():
    features = ["danceability","energy","loudness","liveness","valence","tempo",
    "duration_s","popularity","speechiness","instrumentalness"]

    features2 = ["danceability","energy","loudness","liveness","valence","tempo",
    "duration_s","speechiness","instrumentalness"]

    # load playlist data
    data = load_data()

    # Array of playlist names for indexing and actual playlists
    playlists = data[0]
    names = data[1]

    # the two playlists I care about for this analysis
    noir = playlists[names[9]]
    beast = playlists[names[10]]
    # add purple to see with more samples...
    purple = playlists[names[5]]

    # loop over those two playlists
    for playlist in [noir, beast, purple]:

        # first we need to create a dataframe of the features and their values for the playlist 
        data = {}
        for feature in features:
            data[feature] = playlist[feature]
        df = pd.DataFrame(data, columns=features)

        # create correlation matrix plots
        corrMatrix = df.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
        
        # save the heatmap
        plt.savefig('./images/' + playlist["playlist_name"][0] + '_correlation_matrix.jpeg')

        # create a multiple linear regression for popularity from the other features
        X = df.drop(columns=["popularity"])
        y = df["popularity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
                
        # create a scatter plot of the actual vs predicted values
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Popularity vs. Predicted")
        plt.savefig('./images/' + playlist["playlist_name"][0] + '_popularity_regression.jpeg')

        # print the r squared value for the linear model
        print("R squared value for " + playlist["playlist_name"][0] + ": " + str(reg.score(X_test, y_test)))

        # plot a histogram of the residuals
        plt.figure(figsize=(10, 10))
        plt.hist(y_pred - y_test, bins=20)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residuals")
        plt.savefig('./images/' + playlist["playlist_name"][0] + '_residuals.jpeg')

        # save the linear regression coefficients to a csv file
        df = pd.DataFrame(reg.coef_, columns=["coefficients"])
        df["features"] = features2
        # switch the order of the features and coefficients columns
        df = df[["features", "coefficients"]]
        df.to_csv('./results/' + playlist["playlist_name"][0] + '_regression_coefficients.csv')

if __name__ == '__main__':
    AnalysisOne()
    AnalysisTwo()

    

    


        
