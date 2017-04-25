# -*- coding:utf-8 -*-

import numpy as np
from math import sqrt
import sys

def rmse(x1, x2):
    n = len(x1)
    _sum = 0.0
    for i in range(n):
        _sum += (x1[i] - x2[i]) * (x1[i] - x2[i])

    _sum /= n
    return sqrt(_sum)


def cal_rmse(gt_file, ouput_file):
    output = {}
    point_list = []
    with open(ouput_file, 'r') as f:
        data = f.readlines()
        timestamps = map(int, data[0].rstrip('\n\r').split(',')[1:])
        for point in data[1:]:
            point_data = map(int, point.rstrip('\n\r').split(','))
            output['%s' % point_data[0]] = point_data[1:]
            point_list.append(point_data[0])
    ground_truths = {}

    with open(gt_file, 'r') as f:
        data = f.readlines()
        all_times = map(int, data[0].rstrip('\n\r').split(',')[1:])
        indexes = [all_times.index(p) for p in timestamps]
        all_data = {}
        for point in data[1:]:
            point_data = map(int, point.rstrip('\n\r').split(','))
            all_data['%s' % point_data[0]] = point_data[1:]
        for point in point_list:
            key = '%s' % point
            ground_truths[key] = [all_data[key][i] for i in indexes]

    res = ''
    sum_res = 0.0
    for point in point_list:
        key = '%s' % point
        res += '[%s]: ' % point
        er = rmse(ground_truths[key], output[key])
        res += '%s' % er
        res += '\n'
        sum_res += er * er
    print res
    print '[all]: %s' % (sqrt(sum_res/len(point_list)))




if __name__ == '__main__':
    gt_file = sys.argv[1]
    output = sys.argv[2]
    cal_rmse(gt_file, output)