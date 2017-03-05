# -*- coding:utf-8 -*-


# generate Points.txt
def points_to_point():
    all_points = {}
    with open('graph.csv', 'r') as f:
        for line in f.readlines():
            points = line.replace('\n', '').split(',')
            point_a = points[0]
            point_b = points[1]

            if point_a in all_points:
                __points = all_points[point_a]
                if point_b not in __points:
                    __points.append(point_b)
                all_points[point_a] = __points
            else:
                all_points[point_a] = [point_b]
            if point_b in all_points:
                __points = all_points[point_b]
                if point_a not in __points:
                    __points.append(point_a)
                all_points[point_b] = __points
            else:
                all_points[point_b] = [point_a]

    with open('Points.txt', 'a+') as f:
        for key in all_points:
            line = "%s;" % key
            line += ','.join(all_points[key])
            line += '\n'
            f.write(line)

# get point_count.txt
def get_points_count_list():
    nums = []
    center_point = []
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            point = int(line[0])
            points = line[1].split(',')
            nums.append(len(points)+1)
            center_point.append(point)

    with open('point_count.txt', 'a+') as f:
        for i in range(328):
            line = '%s ' % center_point[i]
            line += '%s\n' % nums[i]
            f.write(line)


def generate_links_split_file(points, point_name):
    data = {}
    with open('speeds_without_zero.csv', 'r') as f:
        for line in f.readlines()[1:]:
            line_elements = map(int, line.replace('\n', '').split(','))
            if line_elements[0] in points or line_elements[0] == point_name:
                data[line_elements[0]] = line_elements[1:]

    with open('speed_data/%s.txt' % point_name, 'a+') as f:
        line = ",".join(['%s' % i for i in data[point_name]])
        line += '\n'
        f.write(line)
        print '<----%s---->' % point_name
        for key in points:
            line = ",".join(['%s' % i for i in data[key]])
            line += '\n'
            f.write(line)

# split the data set
def split_dataset():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            generate_links_split_file(values, title)



# generate predict data

TERM_SIZE = 24

def generate_predicts(points, point_name):
    data = {}
    with open('speeds_without_zero.csv', 'r') as f:
        s = f.readlines()[1:]
        for line in s:
            line_elements = map(int, line.replace('\n', '').split(','))
            if line_elements[0] in points or line_elements[0] == point_name:
                all_points = line_elements[1:]
                max_len = len(all_points)
                data[line_elements[0]] = all_points[max_len-TERM_SIZE:max_len]

    with open('predict_data/%s.txt' % point_name, 'a+') as f:
        line = ",".join(['%s' % i for i in data[point_name]])
        line += '\n'
        f.write(line)
        print '<----%s---->' % point_name
        for key in points:
            line = ",".join(['%s' % i for i in data[key]])
            line += '\n'
            f.write(line)

def get_predict_data():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            generate_predicts(values, title)


if __name__ == '__main__':
    points_to_point()
    get_points_count_list()
    split_dataset()
    get_predict_data()
