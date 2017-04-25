#! /usr/bin/python
import sys


def split_data():
    index = []
    data = {}
    with open('speeds_without_zero.csv', 'r') as f:
        lines = f.readlines()
        title_line = lines[0]
        title_line = title_line.rstrip('\n\r').split(',')[1:]
        times = map(int, title_line)
        # 201603010020
        for time_num in times:
            time_value = time_num % 10000
            if time_value <= 1000 and time_value >= 600:
                index.append(times.index(time_num))
        for line in lines[1:]:
            values = map(int, line.rstrip('\n\r').split(','))
            point = values[0]
            values = values[1:]
            data['%s' % point] = [values[x] for x in index]
    with open('new_speeds.csv', 'a+') as f:
        for key in data:
            line = "%s," % key
            line += ','.join(['%s' % x for x in data[key]])
            line += '\n'
            f.write(line)

# split_data()
"""
get a new csv file from `from_date` to `to_date`
"""
def split_csv_data(input_file, from_date, to_date, output_name):
    data = {}
    timestamps = []
    with open(input_file, 'r') as f:
        s = f.readlines()
        timestamps = s[0].rstrip('\n\r').split(',')[1:]
        timestamps = map(int, timestamps)
        for line in s[1:]:
            line = map(int, line.rstrip('\n\r').split(','))
            data['%s' % line[0]] = line[1:]
    from_index = timestamps.index(from_date)
    end_index = timestamps.index(to_date)
    with open(output_name, 'w+') as f:
        ran = range(from_index, end_index+1, 1)
        line = 'id,'
        line += ','.join(['%s' % timestamps[x] for x in ran])
        line += '\n'
        f.write(line)
        for point in data:
            line = '%s,' % point
            line += ','.join(['%s' % data[point][p] for p in ran])
            line += '\n'
            f.write(line)


def split_by_remove_some_timestamps(input_file, dates, output_name):
    data = {}
    timestamps = []
    with open(input_file, 'r') as f:
        s = f.readlines()
        timestamps = s[0].rstrip('\n\r').split(',')[1:]
        timestamps = map(int, timestamps)
        for line in s[1:]:
            line = map(int, line.rstrip('\n\r').split(','))
            data['%s' % line[0]] = line[1:]
    indexes = range(0, len(timestamps),1)
    print len(indexes)
    for _date in dates:
        from_index = timestamps.index(_date[0])
        end_index = timestamps.index(_date[1])
        for i in range(from_index, end_index+1):
            indexes.remove(i)
        # new_indexes = indexes[0:from_index]
        # new_indexes.extend(indexes[end_index+1:])
    # print indexes
    print len(indexes)
    with open(output_name, 'w+') as f:
        line = 'id,'
        line += ','.join(['%s' % timestamps[x] for x in indexes])
        line += '\n'
        f.write(line)
        for point in data:
            line = '%s,' % point
            line += ','.join(['%s' % data[point][p] for p in indexes])
            line += '\n'
            f.write(line)


def split_out(input_file, dates, ouput_files, output_main_file):
    data = {}
    timestamps = []
    split_timestamps = []
    with open(input_file, 'r') as f:
        s = f.readlines()
        timestamps = s[0].rstrip('\n\r').split(',')[1:]
        timestamps = map(int, timestamps)
        for line in s[1:]:
            line = map(int, line.rstrip('\n\r').split(','))
            data['%s' % line[0]] = line[1:]
    indexes = range(0, len(timestamps),1)
    print len(indexes)
    for _date in dates:
        from_index = timestamps.index(_date[0])
        end_index = timestamps.index(_date[1])
        for i in range(from_index, end_index+1):
            indexes.remove(i)
        split_timestamps.append(range(from_index, end_index+1))

    with open(output_main_file, 'w+') as f:
        line = 'id,'
        line += ','.join(['%s' % timestamps[x] for x in indexes])
        line += '\n'
        f.write(line)
        for point in data:
            line = '%s,' % point
            line += ','.join(['%s' % data[point][p] for p in indexes])
            line += '\n'
            f.write(line)
    i = 0
    for rans in split_timestamps:
        with open(ouput_files[i], 'w+') as f:
            line = 'id,'
            line += ','.join(['%s' % timestamps[x] for x in rans])
            line += '\n'
            f.write(line)
            for point in data:
                line = '%s,' % point
                line += ','.join(['%s' % data[point][p] for p in rans])
                line += '\n'
                f.write(line)

        i += 1





def get_test_data(test_file_name, train_data_name, inputfile, dates):
    data = {}
    timestamps = []
    split_timestamps = []
    with open(inputfile, 'r') as f:
        s = f.readlines()
        timestamps = s[0].rstrip('\n\r').split(',')[1:]
        timestamps = map(int, timestamps)
        for line in s[1:]:
            line = map(int, line.rstrip('\n\r').split(','))
            data['%s' % line[0]] = line[1:]
    indexes = range(0, len(timestamps),1)
    print len(indexes)
    for _date in dates:
        from_index = timestamps.index(_date[0])
        end_index = timestamps.index(_date[1])
        for i in range(from_index, end_index+1):
            indexes.remove(i)
        split_timestamps.extend(range(from_index, end_index+1))

    with open(train_data_name, 'w+') as f:
        line = 'id,'
        line += ','.join(['%s' % timestamps[x] for x in indexes])
        line += '\n'
        f.write(line)
        for point in data:
            line = '%s,' % point
            line += ','.join(['%s' % data[point][p] for p in indexes])
            line += '\n'
            f.write(line)
    with open(test_file_name, 'w+') as f:
        line = 'id,'
        line += ','.join(['%s' % timestamps[x] for x in split_timestamps])
        line += '\n'
        f.write(line)
        for point in data:
            line = '%s,' % point
            line += ','.join(['%s' % data[point][p] for p in split_timestamps])
            line += '\n'
            f.write(line)


# if __name__ == '__main__':
#     filename = 'speeds.csv'
#     # split_csv_data(filename,201603020000,201603022355, '20160302')
#     # split_by_remove_some_timestamps(filename, [(201603050000, 201603062355),
#     #                                            (201603120000, 201603132355),
#     #                                            (201603190000, 201603202355),
#     #                                            (201603260000, 201603272355),
#     #                                            (201604020000, 201604042355),
#     #                                            (201604090000, 201604102355),
#     #                                            (201604160005, 201604172355)],
#     #                                 'speed_nzero.csv')
#
#     # split_out(filename, [(201603110600,201603111000),
#     #                      (201603180600,201603181000),
#     #                      (201604190600,201604191000)],
#     #           ['VadiationSet/311_6_10.csv','VadiationSet/318_6_10.csv','VadiationSet/419_6_10.csv'],'speed_no_valid.csv')
#     #
#     # get_test_data('test_speeds.csv', 'train_speeds.csv', 'speed_nzero.csv',[(201603140000, 201603152355),
#     #                                                                         (201603210000, 201603242355),
#     #                                                                         (201604180000, 201604192355)])
#
def generate_point_list_for_node(num):
    with open('point_count_list_2', 'r') as f:
        data = f.readlines()
        data_len = len(data)
        per_num = data_len/num + 1
        for i in range(0,num-1):
            line = ''.join(data[i*per_num:(i+1)*per_num])
            with open('node%s.train.list' % (i+3), 'w+') as node_file:
                node_file.write(line)

        line = ''.join(data[(num-1)*per_num:])
        with open('node%s.train.list' % (num+2), 'w+') as node_file:
            node_file.write(line)

if __name__ == '__main__':
    num = int(sys.argv[1])
    generate_point_list_for_node(num)

