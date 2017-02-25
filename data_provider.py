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


# def initialize(settings, num, **kwargs):
#     settings.pool_size = sys.maxint
#     #settings.input_types = dense_vector(NODE_NUM)
#     # for
#     inputs = {}
#     for i in range(num):
#         key = 'data_%s' % i
#         inputs[key] = integer_value_sequence(12)
#     inputs['label'] = integer_value(1)
#
#     #integer_value(1)
#
#     settings.input_types = inputs
#     # for i in range(num):
#     #     key = 'data_%s' % i
#     #
#     #     settings.input_types[key] = integer_value_sequence(12)
#     # settings.input_types['label'] = integer_value(1)#dense_vector(4)

TERM_SIZE = 12
NODE_NUM = 3


def initialize(settings, num, **kwargs):
    input_types = []
    for i in range(num):
        input_types.append(dense_vector(12))
    input_types.append(dense_vector(4))
    settings.input_types = input_types


def get_label_value(raw):
    if raw == 1 or raw == 0:
        return [0, 0, 0, 1]
    elif raw == 2:
        return [0, 0, 1, 0]
    elif raw == 3:
        return [0, 1, 0, 0]
    elif raw == 4:
        return [1, 0, 0, 0]


@provider(init_hook=initialize)
def process(settings, filename):
    with open(filename, 'r') as f:
        data = []
        max_len = 0
        for line in f.readlines():
            element = line.replace('\n', '').split(';')[1]
            speeds = map(float, element.split(','))
            data.append(speeds)
            max_len = len(speeds)

        for i in range(max_len-TERM_SIZE-1):
            result = []
            for j in range(NODE_NUM):
                result.append(data[j][i:i+TERM_SIZE])
            result.append(get_label_value(data[0][i+TERM_SIZE]))
            yield result

# @provider(input_types={
#     'data': dense_vector(12),
#     'label': dense_vector(4)
# }, cache=CacheType.CACHE_PASS_IN_MEM)
# def process(settings, filename):
#     with open(filename, 'r') as f:
#         first_line = f.next()
#         elements = first_line.replace('\n', '').split(';')
#         traffic_values = map(int, elements[1].split(','))
#         max_len = len(traffic_values)
#         for i in range(max_len-TERM_SIZE-1):
#             yield traffic_values[i:i + TERM_SIZE], get_label_value(traffic_values[i + TERM_SIZE])

    # with open(filename, 'r') as f:
    #     data = []
    #     max_len = 0
    #     for line in f.readlines():
    #         elements = line.replace('\n', '').split(';')
    #         traffic_values = map(int, elements[1].split(','))
    #         data.append(traffic_values)
    #         max_len = len(traffic_values)
    #
    #     settings.logger.info('data')
    #     for i in range(max_len-TERM_SIZE-1):
    #         result = {}
    #         # for j in range(NODE_NUM):
    #         #     key = 'data_%s' % i
    #         #     result[key] = data[j][i:i+TERM_SIZE]
    #         # # label_raw_value = data[0][i+TERM_SIZE]
    #         # # label = get_label_value(label_raw_value)
    #         # result['label'] = data[0][i+TERM_SIZE]
    #         # yield result
    #         yield data[0][i:i+TERM_SIZE], get_label_value(data[0][i+TERM_SIZE])
    #


# def test_process():
#     with open('data/test.txt', 'r') as f:
#         data = []
#         max_len = 0
#         for line in f.readlines():
#             element = line.replace('\n', '').split(';')[1]
#             speeds = map(float, element.split(','))
#             data.append(speeds)
#             max_len = len(speeds)
#
#         for i in range(max_len-TERM_SIZE-1):
#             result = []
#             for j in range(NODE_NUM):
#                 result.append(data[j][i:i+TERM_SIZE])
#             result.append(data[0][i+TERM_SIZE])
#             yield result[0:NODE_NUM]
# s = test_process()
# print s.next()
# print s.next()