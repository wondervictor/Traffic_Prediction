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


def get_points_count_list_2():
    nums = []
    with open('two_dist_point', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            point = int(line[0])
            nearby = map(int, line[1].split(','))
            sub = map(int, line[2].split(','))
            nums.append((point, len(nearby), len(sub)))
    with open('point_count_list_2', 'w+') as f:
        for tp in nums:
            line = '%s ' % tp[0]
            line += '%s ' % tp[1]
            line += '%s\n' % tp[2]
            f.write(line)


def generate_links_split_file(points, point_name):
    data = {}
    # speeds_without_zero
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


def generate_speed_data_2(center_point, subnodes, nearby,data_set, file_dir):
    center = data_set['%s' % center_point]
    nearby_nodes = []
    for i in nearby:
        nearby_nodes.append(data_set['%s' % i])
    subs = []
    for i in subnodes:
        subs.append(data_set['%s' % i])

    with open('%s/%s.txt' % (file_dir, center_point), 'w+') as f:
        line = ','.join(['%s' % x for x in center])
        line += '\n'
        f.write(line)
        print '<----%s---->' % center_point
        for node in nearby_nodes:
            line = ",".join(['%s' % i for i in node])
            line += '\n'
            f.write(line)

        for node in subs:
            line = ",".join(['%s' % i for i in node])
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
    # speeds_without_zero
    with open('speeds_without_zero.csv', 'r') as f:
        s = f.readlines()[1:]
        # s = f.readlines()
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





def create_dataset(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            line = map(int, line.rstrip('\r\n').split(','))
            data['%s' % line[0]] = line[1:]
    return data


def generate_predicts_2(center_point, nearby, subnodes,data_set):

    center = data_set['%s' % center_point]
    max_len = len(center)
    center = center[max_len-TERM_SIZE:max_len]
    nearby_nodes = []
    for i in nearby:
        nearby_nodes.append(data_set['%s' % i][max_len-TERM_SIZE:max_len])

    subs = []
    for i in subnodes:
        subs.append(data_set['%s' % i][max_len-TERM_SIZE:max_len])

    with open('predict_data/%s.txt' % center_point, 'w+') as f:
        line = ','.join(['%s' % x for x in center])
        line += '\n'
        f.write(line)
        print '<----%s---->' % center_point
        for node in nearby_nodes:
            line = ",".join(['%s' % i for i in node])
            line += '\n'
            f.write(line)

        for node in subs:
            line = ",".join(['%s' % i for i in node])
            line += '\n'
            f.write(line)


def generate_predicts_for_validation(center_point, nearby, subnodes, data_set):

        center = data_set['%s' % center_point]
        max_len = len(center)
        center = center[0:24]
        nearby_nodes = []
        for i in nearby:
            nearby_nodes.append(data_set['%s' % i][0:24])
        subs = []
        for i in subnodes:
            subs.append(data_set['%s' % i][0:24])

        with open('predict_data/%s.txt' % center_point, 'w+') as f:
            line = ','.join(['%s' % x for x in center])
            line += '\n'
            f.write(line)
            print '<----%s---->' % center_point
            for node in nearby_nodes:
                line = ",".join(['%s' % i for i in node])
                line += '\n'
                f.write(line)

            for node in subs:
                line = ",".join(['%s' % i for i in node])
                line += '\n'
                f.write(line)


# point.txt
def get_predict_data():
    with open('Points.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n\r').split(';')
            title = int(line[0])
            values = map(int, line[1].split(','))
            generate_predicts(values, title)

# point_copu.txt
def get_predict_data_2(dataset):
    with open('two_dist_point', 'r') as f:
        for line in f.readlines():
            part = line.rstrip('\n\r').split(';')
            title = int(part[0])
            nearby_nodes = map(int, part[1].split(','))
            subnodes = map(int, part[2].split(','))
            generate_predicts_2(title, nearby_nodes, subnodes, dataset)


def get_speed_data_2(dataset, file_dir):
    with open('two_dist_point', 'r') as f:
        for line in f.readlines():
            part = line.rstrip('\n\r').split(';')
            title = int(part[0])
            nearby_nodes = map(int, part[1].split(','))
            subnodes = map(int, part[2].split(','))
            generate_speed_data_2(title, nearby_nodes, subnodes, dataset, file_dir)


def get_predict_valid(dataset):
    with open('two_dist_point', 'r') as f:
        for line in f.readlines():
            part = line.rstrip('\n\r').split(';')
            title = int(part[0])
            nearby_nodes = map(int, part[1].split(','))
            subnodes = map(int, part[2].split(','))
            generate_predicts_for_validation(title, nearby_nodes, subnodes, dataset)


import split_data

if __name__ == '__main__':

    points_to_point()
    get_points_count_list_2()
    # split_dataset()
    # get_predict_data()

    # dataset = create_dataset('speeds_without_zero.csv')
    # get_points_count_list_2()
    # get_speed_data_2(dataset)
    # get_predict_data_2(dataset)
    # dataset = create_dataset('VadiationSet/419_6_10.csv')
    # get_predict_valid(dataset)
    # dataset = create_dataset('test_speeds.csv')
    # get_speed_data_2(dataset, 'test')
    # dataset = create_dataset('train_speeds.csv')
    # get_speed_data_2(dataset, 'train')
    #dataset = create_dataset('speeds.csv')


    # no zero
    # split_data.split_by_remove_some_timestamps('speeds.csv',
    #                                            [(201603050000, 201603062355),
    #                                             (201603120000, 201603132355),
    #                                             (201603190000, 201603202355),
    #                                             (201603260000, 201603272355),
    #                                             (201604020000, 201604042355),
    #                                             (201604090000, 201604102355),
    #                                             (201604160005, 201604172355)],
    #                                            'speed_nzero.csv')
    # filename = 'speed_nzero.csv'
    #
    # # validation
    # split_data.split_out(filename, [(201603110600,201603111000),
    #                                 (201603180600,201603181000),
    #                                 (201604190600,201604191000)],
    #                     ['VadiationSet/311_6_10.csv', 'VadiationSet/318_6_10.csv', 'VadiationSet/419_6_10.csv'], 'speed_no_valid.csv')
    #
    # split_data.get_test_data('test_speeds.csv', 'train_speeds.csv', 'speed_no_valid.csv', [(201603140000, 201603152355),
    #                                                                                        (201603210000, 201603242355),
    #                                                                                        (201604180000, 201604192355)])
    # dataset = create_dataset('test_speeds.csv')
    # get_speed_data_2(dataset, 'test')
    #
    # dataset = create_dataset('train_speeds.csv')
    # get_speed_data_2(dataset, 'train')
