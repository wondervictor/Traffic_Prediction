# -*- coding:utf-8 -*-

from GraphConverter import *


def split_dataset():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            select_point_to_test(values, title)

split_dataset()