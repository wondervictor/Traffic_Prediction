#! /usr/bin/python

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
    for _date in dates:
        from_index = timestamps.index(_date[0])
        end_index = timestamps.index(_date[1])
        new_indexes = indexes[0:from_index]
        new_indexes.extend(indexes[end_index+1:])
        indexes = new_indexes
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

if __name__ == '__main__':
    filename = 'speeds.csv'
    # split_csv_data(filename,201603020000,201603022355, '20160302')
    split_by_remove_some_timestamps(filename, [(201603050000, 201603062355),
                                               (201603120000, 201603132355),
                                               (201603190000, 201603202355),
                                               (201603260000, 201603272355),
                                               (201604020000, 201604042355),
                                               (201604090000, 201604102355),
                                               (201604160005, 201604172355)],
                                    'speed_nzero.csv')





