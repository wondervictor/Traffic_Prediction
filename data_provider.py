from paddle.trainer.PyDataProvider2 import *
import re
import logging
import random

TERM_SIZE = 24


def initialize(settings, num, point, **kwargs):
    s = dict()
    settings.point = point
    settings.num = num
    for i in range(num):
        key = 'data_%s' % i
        s[key] = dense_vector_sequence(TERM_SIZE)

    #s['label'] = integer_value_sequence(TERM_SIZE)
    for i in range(TERM_SIZE):
        label_key = 'label_%s' % i
        s[label_key] = integer_value(4)
    settings.input_types = s


@provider(init_hook=initialize, cache=CacheType.NO_CACHE,should_shuffle=True)
def process(settings, filename):
    data = []
    node_num = settings.num
    max_len = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            speed = map(int, line.rstrip('\n').split(','))
            data.append(speed)
            max_len = len(speed)

    for i in range(0,max_len-2*TERM_SIZE, TERM_SIZE):
        result = dict()
        for j in range(node_num):
            key = 'data_%s' % j
            result[key] = [[data[j][k] for k in range(i, i+TERM_SIZE)]]

        # result['label'] = data[0][i+TERM_SIZE] - 1
        # if result['label'] == -1:
        #     result['label'] = random.randint(0, 3)

        label = data[0][i+TERM_SIZE:i+2*TERM_SIZE]
        for j in range(TERM_SIZE):
            if label[j] == 0:
                 label[j] = random.randint(1, 4)
            label[j] -= 1
            label_key = 'label_%s' % j
            result[label_key] = label[j]
        # logging.info(result)
        # result['label'] = label
        yield result





# def initialize(settings, num, point, **kwargs):
#     s = dict()
#     settings.point = point
#     settings.num = num
#     for i in range(num):
#         key = 'data_%s' % i
#         s[key] = dense_vector_sequence(TERM_SIZE)
#     for i in range(TERM_SIZE):
#         s['label_%s' % i] = integer_value(4)
#     settings.input_types = s
#
#
# @provider(init_hook=initialize,cache=CacheType.CACHE_PASS_IN_MEM)
# def process(settings, filename):
#     data = []
#
#     max_len = 0
#     node_num = settings.num
#
#     with open(filename, 'r') as f:
#         for line in f.readlines():
#             speeds = map(int, line.rstrip('\n').split(','))
#             data.append(speeds)
#             max_len = len(speeds)
#         for i in range(max_len - 2*TERM_SIZE - 1):
#             result = dict()
#
#             for j in range(node_num):
#                 key = 'data_%s' % j
#                 result[key] = [[data[j][k] - 1 for k in range(i, i + TERM_SIZE)]]
#             labels = data[0][i+TERM_SIZE:i+TERM_SIZE*2]
#             if 0 in labels:
#                 continue
#             for p in range(TERM_SIZE):
#                 key = 'label_%s' % p
#                 result[key] = labels[p]-1
#             yield result
#
#
def predict_initialize(settings, num, point, **kwargs):
    s = dict()
    settings.point = point
    settings.num = num
    for i in range(num):
        key = 'data_%s' % i
        s[key] = dense_vector_sequence(TERM_SIZE)
    settings.input_types = s


@provider(init_hook=predict_initialize,cache=CacheType.CACHE_PASS_IN_MEM)
def process_predict(settings, filename):
    with open(filename,'r') as f:
        data = []
        max_len = 0
        node_num = settings.num
        result = dict()
        for line in f.readlines():
            speeds = map(int, line.rstrip('\n').split(','))
            data.append(speeds)
            max_len = len(speeds)
        for i in range(node_num):
            key = 'data_%s' % i
            result[key] = [[data[i][k] - 1 for k in range(0, TERM_SIZE)]]
        yield result
