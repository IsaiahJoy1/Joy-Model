import os
import SpotifyAPIHelper as sh
from dotenv import load_dotenv
import json

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

session = sh.authenticate(CLIENT_ID, CLIENT_SECRET)

# Insert playlist below 
my_data = sh.get_data_for_ML(session, '')
sh.add_ML_lables_by_playlist(session,'', my_data)

with open('data1.json', 'w') as fp:
    json.dump(my_data, fp)