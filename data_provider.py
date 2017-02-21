from paddle.trainer.PyDataProvider2 import *
import numpy as np

# def init_hook(settings, **kwargs):
#     pass


def normalize(x):
    max_num = 0
    normalized_nums = []
    for num in x:
        max_num = max(max_num, num)
    for num in x:
        normalized_nums.append(float(num/max_num))
    return normalized_nums


@provider(input_types={
    'node': dense_vector(328),
    'label': dense_vector(328)
})
def process(settings, filename):
    with open(filename, 'r') as f:
        f.next()
        nodes = []
        for num, line in enumerate(f):
            nodes.append(map(int, line.rstrip('\r\n').split(",")[0]))

        nodes = normalize(nodes)
        all_speeds = []
        num_speed = 0
        for row_num, line in enumerate(f):
            speeds = map(int, line.rstrip('\r\n').split(",")[1:])
            num_speed = len(speeds)
            all_speeds.append(speeds)

        i = 0
        while i < num_speed:
            i += 1
            yield {
                'node': nodes,
                'label': [all_speeds[x][i] for x in range(328)]
            }
