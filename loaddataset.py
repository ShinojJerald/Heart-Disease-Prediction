# -*- coding: utf-8 -*-

import pymongo as pm

connection = pm.MongoClient()
heart = connection.heart
heart_collection = heart.heart_data
heart_collection.drop()

def import_content(filename):
    print('Reading file...')
    data = open(filename, 'r')
    header = True
    keys = []
    step = 0
    batch_data = []
    for line in data:
        line_data = {}
        if header:
            keys = line.split(',')
            print(keys)
            header = False
        else:
            line = line.split(',')
            for k in range(len(keys)):
                line_data[keys[k]] = line[k]
            if step % 100000 == 0:
                batch_data = load_many_and_report(heart_collection, batch_data, step)
            else:
                batch_data += [line_data]
        step += 1
    batch_data = load_many_and_report(heart_collection, batch_data, step)
    print('Finished loading data.')
def load_many_and_report(coll, batch_data, step):
    coll.insert_many(batch_data)
    print('%d lines loaded...' % step)
    batch_data = []
    return batch_data
import_content("/Users/jerald/Desktop/GMU/AIT-614/project/heart.txt")