# -*- coding:utf-8 -*-

from GraphConverter import *


def split_dataset():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            select_point_to_test(values, title)


def find_miss_data():
    all_points = []
    with open('point_count.txt', 'r') as f:
        s = f.readlines()
        for i in s:
            i = map(int, i.rstrip('\r\n').split(' '))
            all_points.append(i[0])
        print len(all_points)

    values = []
    miss_values = []
    with open('miss_data', 'r') as f:
        s = f.readlines()
        for i in s:
            i = map(int, i.rstrip('\n\r').split(' '))
            print i[0]
            values.append(i[0])

    for i in all_points:
        if i not in values:
            miss_values.append(i)

    with open('missing', 'a+') as f:
        lines = '\n'.join(['%s' % i for i in miss_values])
        f.write(lines)

# find_miss_data()


def get_missing_point():
    missing_points = []
    with open('missing', 'r') as f:
        s = f.readlines()
        for i in s:
            i = map(int, i.rstrip('\n\r').split(' '))
            print i
            missing_points.append(i[0])
        print len(missing_points)
    data = {}
    with open('point_count.txt') as f:
        s = f.readlines()
        for i in s:
            i = map(int, i.rstrip('\n\r').split(' '))
            data['%s' % i[0]] = i[1]

    with open('missing_points', 'a+') as f:
        for key in missing_points:
            line = '%s %s\n' %(key, data['%s' % key])
            f.write(line)

get_missing_point()