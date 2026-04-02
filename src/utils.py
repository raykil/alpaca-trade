import json

def loadConfig(configpath, mode):
    with open(configpath, 'r') as c: return json.load(c)[mode]
