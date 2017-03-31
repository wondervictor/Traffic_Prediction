from paddle.trainer_config_helpers import *
# from paddle.trainer.config_parser import *
import numpy as np
import logging

is_predict = get_config_arg('is_predict', bool, False)
num = get_config_arg('num', int, 0)
point = get_config_arg('point', int, 0)


with open('data/train.list', 'w') as f:
    f.write('data/speed_data/%s.txt' % point)
with open('data/test.list', 'w') as f:
    f.write('data/speed_data/%s.txt' % point)
process = 'process'
if is_predict:
    process = 'process_predict'
    with open('data/pred.list', 'w') as f:
        f.write('data/predict_data/%s.txt' % point)

test = 'data/test.list'
train = 'data/train.list'
if is_predict:
    train = None
    test = 'data/pred.list'

define_py_data_sources2(
    train_list=train,
    test_list=test,
    module='data_provider',
    obj=process,
    args={
        'num': num,
        'point': point,
    }
)

batch_size = 144

if is_predict:
    batch_size = 1

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer(0.001),
    regularization=L2Regularization(8e-4)
)



TERM_SIZE = 24
NODE_NUM = num

# input
input_data = []
for i in range(NODE_NUM):
    key = "data_%s" % i
    input_data.append(data_layer(name=key, size=TERM_SIZE))

output_values = []


# 1 lstmemory cells
lstm_cells = []

for data in input_data:
    lstm_cell = lstmemory(input=data, gate_act=TanhActivation(), act=ReluActivation(), state_act=TanhActivation())
    lstm_cells.append(lstm_cell)

label = data_layer(name='label', size=TERM_SIZE)

lstm_fc_layer = fc_layer(input=lstm_cells, size=3, act=SigmoidActivation())

simple_lstm_layer = simple_lstm(input=lstm_fc_layer,
                                size=TERM_SIZE,
                                gate_act=TanhActivation(),
                                state_act=TanhActivation(),
                                act=ReluActivation())

lastseq_1_layer = last_seq(input=simple_lstm_layer)

dropout_1_layer = dropout_layer(input=lastseq_1_layer, dropout_rate=0.2)

time_1_output_layer = fc_layer(input=dropout_1_layer, size=4, act=ReluActivation())

time_1_value = fc_layer(input=time_1_output_layer, size=1, act=ReluActivation())

output_values.append(time_1_value)

last_time = time_1_value


def neural_unit(input_values, lstm_seq, i):
    paramAttr = ParameterAttribute(initial_max=1.0, initial_min=-1.0)

    # with mixed_layer(size=len(input_values)) as m:
        # m += identity_projection(input_values)
    #    for layer in input_values:
    #        m += identity_projection(layer)
    recent_layer = fc_layer(input=[input_values[k] for k in range(0, i)], size=i, act=ReluActivation())

    fc_nn_layer = fc_layer(input=[recent_layer, lstm_seq], act=ReluActivation(), size=TERM_SIZE*2)

    dropout_nn_layer = dropout_layer(input=fc_nn_layer, dropout_rate=0.2)

    output_layer = fc_layer(input=dropout_nn_layer, size=4, act=ReluActivation())

    time_value = fc_layer(name='time_%s' % i, input=output_layer, size=1, act=ReluActivation())

    return time_value


time_tmp_value = neural_unit(output_values, lastseq_1_layer, 1)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 2)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 3)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 4)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 5)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 6)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 7)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 8)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 9)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 10)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 11)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 12)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 13)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 14)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 15)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 16)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 17)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 18)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 19)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 20)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 21)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 22)
output_values.append(time_tmp_value)

time_tmp_value = neural_unit(output_values, lastseq_1_layer, 23)
output_values.append(time_tmp_value)


out_layer = fc_layer(input=output_values, size=TERM_SIZE, act=ReluActivation())

cost = regression_cost(input=out_layer, label=label)

outputs(cost)




# for i in range(1, TERM_SIZE):
#
#     paramAttr = ParameterAttribute(initial_max=1.0, initial_min=-1.0)
#
#     recent_layer = fc_layer(input=output_values, size=TERM_SIZE, act=ReluActivation())
#
#     fc_nn_layer = fc_layer(input=[recent_layer, lastseq_1_layer], act=ReluActivation(), size=TERM_SIZE*2)
#
#     dropout_nn_layer = dropout_layer(input=fc_nn_layer, dropout_rate=0.2)
#
#     output_layer = fc_layer(input=dropout_nn_layer, size=4, act=ReluActivation())
#
#     time_value = fc_layer(name='time_%s' % i,input=output_layer, size=1, act=ReluActivation())
#
#     new_output_values = output_values
#     new_output_values.append(time_value)
#     output_values = new_output_values


    #output_values.append(time_value)

# with mixed_layer(size=TERM_SIZE) as m:
#     for ly in output_values:
#         m += ly




