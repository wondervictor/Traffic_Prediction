# -*- coding: utf-8 -*-
import sys

def add_timestamp(filename, start, step, num):
    timestamps = []
    for i in range(num):
        time = i * 5
        if time % 60 == 0 and time > 0 :
            start += 100

        timestamps.append(time % 60 + start)
    with open(filename, 'w+') as f:
        line = 'id,'
        line += ','.join(['%s' % s for s in timestamps])
        line += '\n'
        f.write(line)


if __name__ == '__main__':
    name = sys.argv[1]
    start = int(sys.argv[2])
    step = int(sys.argv[3])
    nums = int(sys.argv[4])
    add_timestamp(name, start, step, nums)
