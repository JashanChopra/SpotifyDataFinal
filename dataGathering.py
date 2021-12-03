# importing the necessary packages
import pandas as pd 
import spotipy 
import spotipy.util as util

from spotipy.oauth2 import SpotifyClientCredentials

def getSpotifyToken():
    # credentials for spotify
    sp = spotipy.Spotify() 

    # setting up authorization
    cid ="f395b1d0ba9b4f9f9d83e833bf1267ff" 
    secret = "e68f3c64c84648a594e00e4b95801420"
    redirect_uri ="http://localhost:7777/callback"
    username = "jashanxchopra@gmail.com"

    # scope to get the user library
    scope = 'user-library-read'
    
    # get the spotify token
    token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

    # also grab Spotify client creds
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    return token, sp

def call_playlist(creator, playlist_id, sp):
    # gets information from a specific playlist
    # sourced from: https://www.linkedin.com/pulse/extracting-your-fav-playlist-info-spotifys-api-samantha-jones

    # setup dataframe
    features = ["artist","album","track_name","track_id",
                "danceability","energy","key","loudness",
                "mode", "speechiness","instrumentalness",
                "liveness","valence","tempo", "duration_ms",
                "time_signature", "acousticness"]

    playlist_df = pd.DataFrame(columns = features)
    
    # get the track details
    playlist = sp.user_playlist_tracks(creator, playlist_id)
    tracks = playlist['tracks']['items']
    for track in tracks:
        playlist_features = {}

        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]

        # add the track's popularity
        playlist_features["popularity"] = track["track"]["popularity"]
        
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in features[4:]:
            playlist_features[feature] = audio_features[feature]

        # add the playlist name 
        playlist_features["playlist_name"] = playlist["name"]

        # convert duration_ms to seconds for plotting purposes later
        playlist_features["duration_s"] = playlist_features["duration_ms"]/1000

        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df

if __name__ == '__main__':
    # get spotify token
    token, sp = getSpotifyToken()

    # use tokedef call_playlist(creator, playlist_id, sp):n to aquire playlist information
    yellow = call_playlist("JashanChopra", "1Y9jEeQLlt46VFEeYzZ3zn?si=ef4e02410b1849cd", sp)
    blue = call_playlist("JashanChopra", "5g5x0aWpm2Uh5Nc0JbpIhf?si=47a4ec21ea2d4505", sp)
    green = call_playlist("JashanChopra", "3sx2rV67D7Ro4e6RgO0Amd?si=89ebf3803b184115", sp)  
    red = call_playlist("JashanChopra", "1sA0ICkMKvRf5ghOPJ4Buq?si=bca99e339c294581", sp)  
    black = call_playlist("JashanChopra", "0ilppVom6odKC1YhtlaOX4?si=ff531d8a78034799", sp)  
    purple = call_playlist("JashanChopra", "4apY0heTn8caTtJgyYE188?si=e5d341fccebd4cfd", sp) 

    # get spotify playlists of similar ideas
    movies = call_playlist("Spotify", "37i9dQZF1DX4OzrY981I1W?si=2c3b19568cdb4761", sp)         # compare to yellow
    goldenhour = call_playlist("Spotify", "37i9dQZF1DWUE76cNNotSg?si=76c7b19eb0cf45a9", sp)     # compare to blue
    surfrock = call_playlist("Spotify", "37i9dQZF1DWYzpSJHStHHx?si=40e993be4845415e", sp)       # compare to green
    workout = call_playlist("Spotify", "37i9dQZF1DX76Wlfdnj7AP?si=a473200f0c4a46d2", sp)        # compare to red
    dark = call_playlist("Spotify", "37i9dQZF1DX9LT7r8qPxfa?si=8ee35e5d7a3549c6", sp)           # compare to black
    idk = call_playlist("Spotify", "37i9dQZF1DX59NCqCqJtoH?si=487debe7b74048c1", sp)            # compare to purple

    # export to .csv for later ease of use
    data = [yellow, blue, green, red, black, purple, movies, goldenhour, surfrock, workout, dark, idk]
    names = ["yellow", "blue", "green", "red", "black", "purple", "movies", "goldenhour", "surfrock", "workout", "dark", "idk"]
    for idx, playlist  in enumerate(data):
        playlist.to_csv("./data/playlist_" + names[idx] + ".csv")
