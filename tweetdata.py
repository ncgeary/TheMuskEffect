import json
import tweets.json
import pandas as pd

data = pd.read_json(tweets.json)

print(data)