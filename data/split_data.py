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

split_data()


