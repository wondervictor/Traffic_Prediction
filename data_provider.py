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




@provider(input_types={
    'time': dense_vector(288),
    'label': dense_vector(288),
})
def process(settings, filename):
    with open(filename, 'r') as f:
        f.next()
        pass
    
