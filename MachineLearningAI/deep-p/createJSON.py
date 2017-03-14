import sys
import os
import json


gameObject = {'id': 1, 'player_1': 1, 'player_2': 2}

with open('gameJSON.txt', 'w+') as outfile:
	json.dump(gameObject, outfile)