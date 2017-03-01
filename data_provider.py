from paddle.trainer.PyDataProvider2 import *
import re
import logging

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
# #NODE_NUM = 10
# INPUT_SIZE = 1276
# NODE_NUM = 328


def initialize(settings, num, point, **kwargs):
    s = dict()
    settings.point = point
    settings.num = num
    for i in range(num):
        key = 'data_%s' % i
        s[key] = dense_vector_sequence(12)
    s['label'] = integer_value(4)
    settings.input_types = s


def get_label_value(raw):
    if raw == 1 or raw == 0:
        return [0, 0, 0, 1]
    elif raw == 2:
        return [0, 0, 1, 0]
    elif raw == 3:
        return [0, 1, 0, 0]
    elif raw == 4:
        return [1, 0, 0, 0]


#
# @provider(input_types={
#     'data_0': dense_vector(12),
#     'data_1': dense_vector(12),
#     'data_2': dense_vector(12),
#     'label': integer_value(4)
# })
# def process(settings, filename):
#     with open(filename, 'r') as f:
#         s = f.readlines()[1]
#         element = s.replace('\n', '').split(';')[1]
#         speeds = map(int, element.split(','))
#         lens = len(speeds)
#         for i in range(lens-1-12-12):
#             #print speeds[i:i+TERM_SIZE]
#             #yield [x/2 for x in range(12)], 2
#             yield speeds[i:i+TERM_SIZE], speeds[i+1:i+1+TERM_SIZE], speeds[i+2:i+2+TERM_SIZE], speeds[i+2+TERM_SIZE]
#             # yield {
#             #     'data': speeds[i:i+TERM_SIZE],
#             #     'label': speeds[i+TERM_SIZE]
#             # }
#



# @provider(input_types={
#     'data_0': dense_vector_sequence(12),
#     'data_1': dense_vector_sequence(12),
#     'data_2': dense_vector_sequence(12),
#     'data_3': dense_vector_sequence(12),
#     'data_4': dense_vector_sequence(12),
#     'data_5': dense_vector_sequence(12),
#     'data_6': dense_vector_sequence(12),
#     'data_7': dense_vector_sequence(12),
#     'data_8': dense_vector_sequence(12),
#     'data_9': dense_vector_sequence(12),
#     'label':  integer_value(4)
# }, cache=CacheType.CACHE_PASS_IN_MEM)
@provider(init_hook=initialize,cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, filename):
    data = []
    # with open(filename, 'r') as f:
    #     files = f.readlines()
    max_len = 0
    node_num = settings.num
    # file_name = 'data/speed_data/%s.txt' % settings.point
    # #print file_name
    with open(filename, 'r') as f:
        for line in f.readlines():
            speeds = map(int, line.rstrip('\n').split(','))
            data.append(speeds)
            max_len = len(speeds)
        for i in range(max_len - TERM_SIZE - 1):
            result = dict()
            label = data[0][i + TERM_SIZE] - 1
            if label == -1:
                continue
            for j in range(node_num):
                key = 'data_%s' % j
                result[key] = [[data[j][k] - 1 for k in range(i, i + TERM_SIZE)]]
            result['label'] = label
            yield result



    # for file_name in files:
    #     file_name = file_name.rstrip('\n\r')
    #     data = []
    #     with open(file_name, 'r') as f:
    #         for line in f.readlines():
    #             element = line.replace('\n', '').split(';')[1]
    #             speeds = map(int, line.rstrip('\n').split(','))
    #             data.append(speeds)
    #             max_len = len(speeds)









        # data = []
        # max_len = 0
        # for line in f.readlines():
        #     #element = line.replace('\n', '').split(';')[1]
        #     speeds = map(int, line.rstrip('\n').split(','))
        #     data.append(speeds)
        #     max_len = len(speeds)
        #
        # for i in range(max_len-TERM_SIZE-1):
        #     result = []
        #     for j in range(NODE_NUM):
        #         result.append([data[j][k] for k in range(i, i+TERM_SIZE)])
        #     label = data[0][i+TERM_SIZE]-1
        #     if label == -1:
        #         continue
        #     yield {
        #         'data_0': [[data[0][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_1': [[data[1][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_2': [[data[2][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_3': [[data[3][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_4': [[data[4][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_5': [[data[5][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_6': [[data[6][k]-1 for k in range(i, i + TERM_SIZE)]],
        #         'data_7': [[data[7][k] for k in range(i, i + TERM_SIZE)]],
        #         'data_8': [[data[8][k] for k in range(i, i + TERM_SIZE)]],
        #         'data_9': [[data[9][k] for k in range(i, i + TERM_SIZE)]],
        #         'label_0': label
        #     }


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