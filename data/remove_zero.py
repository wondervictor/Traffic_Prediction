#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

targetrange = []
result = []

with open('speeds.csv', 'r') as inputfile:
    reader = csv.reader(inputfile, dialect = 'excel')
    header = next(reader)
    index = 0
    for title in header:
        if title == '201603050000' or title == '201603110000' or title == '201603120000' or title == '201603180000' or title == '201603190000' or title == '201603260000' or title == '201604020000' or title == '201604090000' or title == '201604110000' or title == '201604160000' or title == '201604190000':
            targetrange.append(index)
        if title == '201603070000' or title == '201603120000' or title == '201603140000' or title == '201603190000' or title == '201603210000' or title == '201603280000' or title == '201604050000' or title == '201604110000' or title == '201604120000' or title == '201604180000' or title == '201604200000':
            targetrange.append(index)
        index = index + 1

with open('speeds.csv', 'r') as inputfile:
    reader = csv.reader(inputfile, dialect = 'excel')
    for row in reader:
        count = 12
        while count >= 0:
            for index in range(targetrange[count], targetrange[count + 1]):
                row.pop(targetrange[count])
            count = count - 2
        result.append(row)




with open('speeds_without_zero.csv', 'w') as outputfile:
    writer = csv.writer(outputfile, dialect = 'excel')
    for row in result:
        writer.writerow(row)
