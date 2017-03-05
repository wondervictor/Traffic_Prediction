# -*- coding: utf-8 -*-


def check():
    points = []
    missing_points = []
    with open('data/point_list.txt', 'r') as f:
        for i in f.readlines():
            i = map(int, i.rstrip('\r\n').split(' '))[0]
            points.append(i)

    data = {}
    with open('result.csv', 'r') as f:
        for line in f.readlines():
            line = map(int, line.rstrip('\r\n').split(','))
            point = line[0]
            sub_points = line[1:]
            data['%s' % point] = sub_points

    with open('results.csv', 'a+') as f:
        for point in points:
            key = '%s' % point
            if key in data:
                line = key + ','+','.join(['%s' % x for x in data[key]])+'\n'
                f.write(line)
            else:
                missing_points.append(key)

        print missing_points

check()