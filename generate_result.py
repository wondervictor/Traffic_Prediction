# -*- coding: utf-8 -*-


import sys

def generate(point, file_name):
    data = []
    with open(file_name, 'r') as f:
        line = f.readlines()[0]
        data = map(int, line.rstrip('\r\n;').split(';'))

    with open('result.csv', 'a+') as f:
        line = '%s,' % point
        line += ','.join(['%s' % (x + 1) for x in data])
        f.write(line)

if __name__ == '__main__':
    point = sys.argv[1]
    name = sys.argv[2]
    generate(point, name)

