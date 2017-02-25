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
    inputs = {}
    for i in range(num):
        key = 'data_%s' % i
        inputs[key] = integer_value_sequence(12)
    inputs['label'] = dense_vector(4)

    #integer_value(1)

    settings.input_types = inputs
    # for i in range(num):
    #     key = 'data_%s' % i
    #
    #     settings.input_types[key] = integer_value_sequence(12)
    # settings.input_types['label'] = integer_value(1)#dense_vector(4)

TERM_SIZE = 12
NODE_NUM = 3


def get_label_value(raw):
    if raw == 1 or raw == 0:
        return [0, 0, 0, 1]
    elif raw == 2:
        return [0, 0, 1, 0]
    elif raw == 3:
        return [0, 1, 0, 0]
    elif raw == 4:
        return [1, 0, 0, 0]


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
            label_raw_value = data[0][i+TERM_SIZE]
            label = get_label_value(label_raw_value)
            result['label'] = label
            #data[0][i+TERM_SIZE]
            yield result








