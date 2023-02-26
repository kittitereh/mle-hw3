from pymongo import MongoClient
import pandas as pd
from pprint import pprint


client = MongoClient('mongodb://0.0.0.0:27017/')

db = client["my_db"]

for doc in db.my_data_new.find({}):
    pprint(doc)