import codecs
import json


class MemoryController:
    def __init__(self, filename):
        # if not os.path.isfile(filename):
        with open(filename, 'w+') as outfile:
            json.dump({-1: "boilerplate"}, outfile)
        obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
        self.db = json.loads(obj_text)
        self.filename = filename

    def add_centroid(self, object_id, centroid):
        if not object_id in self.db:
            self.db[object_id] = list()
            self.db[object_id].append(centroid.tolist())
        else:
            self.db[object_id].append(centroid.tolist())

    def close(self):
        json.dump(self.db, codecs.open(self.filename, 'w+',
                  encoding='utf-8'),
                  separators=(',', ':'),
                  indent=4, )
