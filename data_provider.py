from paddle.trainer.PyDataProvider2 import *
import numpy as np


def normalize(x):
    max_num = 0
    normalized_nums = []
    for num in x:
        max_num = max(max_num, num)
    for num in x:
        normalized_nums.append(float(num/max_num))
    return normalized_nums


NODE_NUM = 328
TERM_NUM = 288
LABEL_NUM = 328


@provider(input_types={
    'data': dense_vector(NODE_NUM),#dense_vector([NODE_NUM, TERM_NUM], seq_type=SequenceType.SEQUENCE),
    'label': dense_vector(NODE_NUM)#dense_vector([NODE_NUM, TERM_NUM], seq_type=SequenceType.SEQUENCE),
    },
    cache=CacheType.CACHE_PASS_IN_MEM
)
def process(settings, filename):
    with open(filename, 'r') as f:
        f.next()
        all_speeds = []
        for row_num, line in enumerate(f):
            speeds = map(float, line.rstrip('\r\n').split(",")[1:])
            all_speeds.append(speeds)
        end_time = len(all_speeds[1])

        for i in range(0, end_time-TERM_NUM):
            speeds = []
            labels = []
            for j in range(0, 328):
                speeds.append(all_speeds[j][i])
                labels.append(all_speeds[j][i+TERM_NUM])
            yield {
                'data': speeds,
                'label': labels
            }

