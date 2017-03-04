from paddle.trainer.PyDataProvider2 import *
import re
import logging

def predict_initialize(settings, num, point, **kwargs):
    s = dict()
    settings.point = point
    settings.num = num
    for i in range(num):
        key = 'data_%s' % i
        s[key] = dense_vector_sequence(12)

@provider(init_hook=predict_initialize, should_shuffle=False)
def process_predict(settings, filename):
    with open(filename, 'r') as f:
        data = []
        node_num = settings.num
        result = dict()
        for line in f.readlines():
            speeds = map(int, line.rstrip('\n\r').split(','))
            data.append(speeds)
        for i in range(node_num):
            key = 'data_%s' % i
            result[key] = [[data[i][k] - 1 for k in range(0, 12)]]
        yield result
