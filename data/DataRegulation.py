# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def value_normalization(value):
    return round(value)

def newton(exist_values, missing_values):
    num_exists = len(exist_values)
    arr = np.zeros([num_exists, num_exists])
    for i in range(num_exists):
        arr[i][0] = exist_values[i][1]
    i = 1
    while i < num_exists:
        j = 1
        while j <= i:
            xi = exist_values[i][0]
            xj = exist_values[i-j][0]
            arr[i][j] = (arr[i][j-1]-arr[i-1][j-1])/(xi-xj)
            j += 1
        i += 1
    print arr
    # calculate
    fixed_values = []
    for one in missing_values:
        s = 0
        t = 1
        value = 0.0
        while s < num_exists-1:
            t *= (one-exist_values[s][0])
            value += arr[s+1][s+1] * t
            s += 1
        value += arr[0][0]
        fixed_values.append((one, '%s' % value_normalization(value)))
    return fixed_values

# def newton(exist_values, missing_values):
#     num_exists = len(exist_values)
#     arr = np.zeros([num_exists,num_exists])
#     for i in range(num_exists):
#         arr[i][0] = int(exist_values[i][1])
#
#     i = 1
#     while i < num_exists:
#         j = 1
#         while j <= i:
#             xi = exist_values[i][0]
#             xj = exist_values[i-j][0]
#             arr[i][j] = (arr[i][j]-arr[i-1][j])/(xi-xj)
#             j += 1
#         i += 1
#
#     # calculate
#     fixed_values = []
#     for one in missing_values:
#         s = 0
#         t = 1
#         value = 0
#         while s < num_exists-1:
#             t *= (one-exist_values[s][0])
#             value += arr[s+1][s+1] * t
#             s += 1
#         value += arr[0][0]
#         fixed_values.append((one, value_normalization(value)))
#     return fixed_values

# def newton(exist_values, missing_values):
#     num_exists = len(exist_values)
#     arr = np.zeros([num_exists, num_exists])
#     a = np.zeros(num_exists)
#     y = np.zeros(num_exists)
#     for i in range(num_exists):
#         xi = int(exist_values[i][0])
#         last_value = 1
#         arr[i][0] = 1
#         for j in range(i):
#             xj = int(exist_values[j][0])
#             last_value *= xi-xj
#             arr[i][j+1] = last_value
#     # y
#     for i in range(num_exists):
#         y[i] = exist_values[i][1]
#     a = np.dot(arr**-1, y)
#     print a
#     fixed_values = []
#     for missing_value in missing_values:
#         fix_value = 0
#         fix_value = a[0]
#         num = 0
#         last_value = 1
#         for i in range(num_exists):
#             last_value *= missing_value-exist_values[i][0]
#             fix_value += a[num]*
#




def newton_insert_value(data):
    with open('speeds_fill.csv', 'a+') as f:
        for element in data:
            missing_indexes = []
            exist_values = []
            i = 0
            for one in element:
                if one == '0':
                    missing_indexes.append(i)
                else:
                    value = int(one)
                    exist_values.append((i, value))
                i += 1
            fixed_values = newton(exist_values, missing_indexes)
            new_data = []
            for one in exist_values:
                new_data[one[0]] = one[1]
            for one in fixed_values:
                new_data[one[0]] = one[1]
            line = ','.join(new_data)
            line += '\n'
            f.write(line)


def prepare_data():
    with open('speeds.csv', 'r') as f:
        for line in f.readlines()[1:]:
            nums = line.replace('\n', '').split(',')
            speeds = nums[1:]
            with open('speeds_fill.csv', 'a+') as f:
                missing_indexes = []
                exist_values = []
                i = 0
                for one in speeds:
                    if one == '0':
                        missing_indexes.append(i)
                    else:
                        value = int(one)
                        exist_values.append((i, value))
                    i += 1
                fixed_values = newton(exist_values, missing_indexes)
                new_data = []
                for one in exist_values:
                    new_data[one[0]] = one[1]
                for one in fixed_values:
                    new_data[one[0]] = one[1]
                line = ','.join(new_data)
                line += '\n'
                f.write(line)


prepare_data()

