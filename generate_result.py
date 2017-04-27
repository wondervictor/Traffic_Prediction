# -*- coding: utf-8 -*-


import sys
import csv_timestamp


def generate(point, file_name):
    data = []
    with open(file_name, 'r') as f:
        line = f.readlines()[0]
        data = map(int, line.rstrip('\r\n;').split(';'))

    with open('result.csv', 'a+') as f:
        line = '%s,' % point
        line += ','.join(['%s' % (x + 1) for x in data])
        line += '\n'
        f.write(line)

# if __name__ == '__main__':
#     point = sys.argv[1]
#     name = sys.argv[2]
#     generate(point, name)

if __name__ == '__main__':
    csv_timestamp.add_timestamp('result.csv', 201605250800, 5, 24)
    point_list = []
    with open('data/point_count_list_2', 'r') as f:
        data = f.readlines()
        for point in data:
            point = map(int, point.rstrip('\n\r').split(' '))[0]
            generate(point, 'result/%s/rank-00000' % point)
