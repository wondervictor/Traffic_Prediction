# -*- coding:utf-8 -*-

TERM_SIZE = 24

def generate_predicts(points, point_name):
    data = {}
    with open('speeds_without_zero.csv', 'r') as f:
        s = f.readlines()[1:]
        for line in s:
            line_elements = map(int, line.replace('\n', '').split(','))
            if line_elements[0] in points or line_elements[0] == point_name:
                print '---> %s' % line_elements[0]
                all_points = line_elements[1:]
                max_len = len(all_points)
                data[line_elements[0]] = all_points[max_len-TERM_SIZE:max_len]

    with open('predict_2_data/%s.txt' % point_name, 'a+') as f:
        line = ",".join(['%s' % i for i in data[point_name]])
        line += '\n'
        f.write(line)
        print '<----%s---->' % point_name
        for key in points:
            line = ",".join(['%s' % i for i in data[key]])
            line += '\n'
            f.write(line)
            print key

def get_predict_data():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            generate_predicts(values, title)

get_predict_data()