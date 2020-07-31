import json

def writeJson(data, name):
    with open(name+".json", 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',',':'))

def readJson(name):
    with open(name, 'r', encoding='utf8') as f:
        return json.load(f)
    