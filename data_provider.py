from paddle.trainer.PyDataProvider2 import *
import numpy as np
import sys



def normalize(x):
    max_num = 0
    normalized_nums = []
    for num in x:
        max_num = max(max_num, num)
    for num in x:
        normalized_nums.append(float(num/max_num))
    return normalized_nums




def initialize(settings, num, **kwargs):
    settings.pool_size = sys.maxint
    #settings.input_types = dense_vector(NODE_NUM)
    # for
    for i in range(num):
        key = 'data_%s' % i
        settings.input_types[key] = integer_value_sequence(12)
    settings.input_types['label'] = integer_value(2)

TERM_SIZE = 12
NODE_NUM = 4

@provider(init_hook=initialize, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, filename):
    with open(filename, 'r') as f:
        data = []
        max_len = 0
        for line in f.readlines():
            elements = line.replace('\n', '').split(';')
            traffic_values = map(int, elements[1].split(','))
            data.append(traffic_values)
            max_len = len(traffic_values)

        for i in range(max_len-TERM_SIZE-1):
            result = {}
            for j in range(NODE_NUM):
                key = 'data_%s' % i
                result[key] = data[j][i:i+TERM_SIZE]
            label = data[0][i+TERM_SIZE+1]
            result['label'] = label
            yield result








