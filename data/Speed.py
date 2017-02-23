# -*- coding:utf-8 -*-


# 10178/288 = 255

def get_dim_of_data():
    with open('speeds.csv', 'r') as f:
        s = f.readlines()
        height = len(s)-1
        print height
        first_line = s[1]
        elements = first_line.rstrip('\r\n').split(',')
        width = len(elements) - 1
        return height, width

