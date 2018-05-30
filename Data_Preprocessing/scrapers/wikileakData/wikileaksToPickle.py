
import pandas as pd
import numpy as np
import json
import glob
import pickle
from datetime import datetime


date_to_fromTo_Title_outputPath = "wikileaks_1987-2013_date2idFromToTitle.p"
id_to_tt_outputPath = "wikileaks_1987-2013_id2Text.p"
#fromFilePath = "wikileak/*.json"

class MacOSFile(object):
    
    def __init__(self, f):
        self.f = f
    
    def __getattr__(self, item):
        return getattr(self.f, item)
    
    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)
    
    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

def main():
    date_to_fromTo_title = {}
    id_to_tt = {}
    idx = 0 # assgin unique id for mapping title with text
    for folder in glob.glob("all/*"):
        print("open year")
        for filename in glob.glob(folder + "/*.json"):
            print(filename)
            with open(filename) as json_data:
                idx += 1
                data = json.load(json_data)
                date = data['Date:']
                date = date[:date.find(',')]
                date = datetime.strptime(date, '%Y %B %d')
                print(date)

                cable = []
                cable.append(idx)
                cable.append(data['From:'])
                cable.append(data['To:'])
                cable.append(data['title'])

                id_to_tt.update({idx:[data['title'],data['text']]})

                if date not in date_to_fromTo_title:
                    newList = []
                    newList.append(cable)
                    date_to_fromTo_title.update({date:newList})
                else:
                    date_to_fromTo_title[date].append(cable)

    pickle_dump(date_to_fromTo_title, date_to_fromTo_Title_outputPath)
    pickle_dump(id_to_tt, id_to_tt_outputPath)

if __name__ == '__main__':
	main()
