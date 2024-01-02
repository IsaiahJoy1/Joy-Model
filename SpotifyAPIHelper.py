import re
import pandas as pd
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

import JoyModel as jm
from JoyModel import MLDataStruct

list_of_segments_features = ['start','duration','loudness_start','loudness_max','loudness_max_time','loudness_end','pitches','timbre']
list_of_sections_features = ['start','duration','loudness','tempo','key','mode','time_signature']
list_of_features = ['acousticness','liveness','speechiness','instrumentalness','energy','loudness','danceability','valence','tempo','key','mode']

list_of_segments_features_loudness = ['start','duration','loudness_start','loudness_max','loudness_max_time','loudness_end']
list_of_segments_features_pitches = ['start','duration','pitches','timbre']

### Basic
#TODO Update
def authenticate(CLIENT_ID:str, CLIENT_SECRET:str) -> Spotify:
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # create spotify session object
    # session
    return Spotify(client_credentials_manager=client_credentials_manager)

def get_playlist_song_ids (session:Spotify, PLAYLIST_LINK:str) -> list[str]:
    #validate that link is in proper format
    if match := re.match(r"https://open.spotify.com/playlist/(.*)\?", PLAYLIST_LINK):
        playlist_uri = match.groups()[0]
    else:
        raise ValueError("Expected format: https://open.spotify.com/playlist/...")
    
    # get list of tracks in a given playlist (note: max playlist length 100)
    results = session.playlist_tracks(playlist_uri)

    tracks = [str(track['track']['id']) for track in results['items']]

    while results['next']:
        results = session.next(results)
        tracks.extend(str(track['track']['id']) for track in results['items'])

    return tracks

### Analysis
def get_audio_analysis_data(session:Spotify, song_id, feature, list_of_sub_features):
    #parse some song data after requesting a song
    song = session.audio_analysis(song_id)
    data_list = []
    for i in range(len(song[feature])):
        diction = song[feature][i]
        data =[]
        for sub_feature in list_of_sub_features:
            if isinstance(diction[sub_feature], list):
                data.extend(diction[sub_feature])
            else:
                data.append(diction[sub_feature])
        data_list.append(data)
    return data_list

def get_audio_analysis_data_lists(session:Spotify, song_id:str, features:list[str], list_of_sub_features_list:list[list[str]]) -> dict:
    #parse some song data after requesting a song
    song = session.audio_analysis(song_id)
    song_data = {}
    for i in range(len(features)):
        data_list = []
        list_of_sub_features = list_of_sub_features_list[i]
        feature = features[i]
        for j in range(len(song[feature])):
            diction = song[feature][j]
            data =[]
            for sub_feature in list_of_sub_features:
                if isinstance(diction[sub_feature], list):
                    data.extend(diction[sub_feature])
                else:
                    data.append(diction[sub_feature])
            data_list.append(data)
        song_data[feature] = data_list
    return song_data

def get_list_of_song_analysis(session:Spotify, songs, feature, list_of_sub_features):
    feature_list = []
    maxLen=0
    
    for song_id in songs:
        timp = get_audio_analysis_data(session, song_id, feature, list_of_sub_features)
        feature_list.append(timp)
        
        if len(timp) > maxLen:
            maxLen = len(timp)
        
    return feature_list, maxLen

### Features
def get_audio_feature_data(session:Spotify, song_id:str, features:list[str]):
    #parse some song data after requesting a song
    song = session.audio_features(song_id)[0]
    data =[]
    for feature in features:
        if isinstance(song[feature], list):
            data.extend(song[feature])
        else:
            data.append(song[feature])
    return data

def get_playlist_audio_features (session:Spotify, songs, features):
    feature_list = []

    for song_id in songs:
        timp = get_audio_feature_data(session, song_id, features)
        feature_list.append(timp)

    return feature_list

### Helpfull
def get_playlist_track_analysis(session:Spotify, playlist_link):
    data = {}
    tracks = get_playlist_song_ids(session, playlist_link)

    data['segments'], data['longest_segment'] = get_list_of_song_analysis(session, tracks, 'segments', list_of_segments_features)
    
    data['sections'], data['longest_section'] = get_list_of_song_analysis(session, tracks, 'sections', list_of_sections_features)
    
    data['features'] = get_playlist_audio_features (session, tracks, list_of_features)

    return data

def get_data_for_ML(session:Spotify, playlist_link:str) -> MLDataStruct:
    struct = MLDataStruct()
    tracks = get_playlist_song_ids(session, playlist_link)

    longest_segment = 0
    longest_section = 0

    for track in tracks:
        song = jm.DataStructFromat()

        song_data = get_audio_analysis_data_lists(session, track, ['segments','sections'], [list_of_segments_features,list_of_sections_features])
        
        song['segments'] = song_data['segments']
        if len(song_data['segments']) > longest_segment:
            longest_segment = len(song_data['segments'])

        song['sections'] = song_data['sections']
        if len(song_data['sections']) > longest_section:
            longest_section = len(song_data['sections'])

        song['features'] = get_audio_feature_data(session, track, list_of_features)

        struct.data.append(song)

    struct.notes.append(jm.DataNote("segments", "RNN" , longest_segment))
    struct.notes.append(jm.DataNote("sections", "RNN" , longest_section))
    struct.notes.append(jm.DataNote("features", "NN" , None))

    return struct
        
def get_labes_by_playlist(session:Spotify, playlist_link:str, track_list, yes=1, no=0, labels=None):
    if labels == None:
        labels = [no] * len(track_list)

    playlist = get_playlist_song_ids (session, playlist_link)

    for i in range(len(track_list)):
        if track_list[i] in playlist:
            labels[i] = yes

    return labels

def add_ML_lables_by_playlist(session:Spotify, playlist_link:str, MLdata:MLDataStruct, yes:any=1, no:any=0) -> None:
    playlist = get_playlist_song_ids (session, playlist_link)

    for i in range(len(MLdata['data'])):
        if MLdata['data'][i]['id'] in playlist:
            MLdata['data'][i]['label'] = yes
        elif no != None:
            MLdata['data'][i]['label'] = no
