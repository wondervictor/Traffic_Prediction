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

batch_size = 128

if is_predict:
    batch_size = 1

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=MomentumOptimizer(0.0001),
    regularization=L2Regularization(8e-4)
)



TERM_SIZE = 24
NODE_NUM = num

# input
input_data = []
for i in range(NODE_NUM):
    key = "data_%s" % i
    input_data.append(data_layer(name=key, size=TERM_SIZE))

output = []
cost_output = []


# 1 lstmemory cells
lstm_cells = []

for data in input_data:
    lstm_cell = lstmemory(input=data, gate_act=TanhActivation(), act=ReluActivation(), state_act=TanhActivation())
    lstm_cells.append(lstm_cell)

lstm_fc_layer = fc_layer(input=lstm_cells, size=3, act=SigmoidActivation())

simple_lstm_layer = simple_lstm(input=lstm_fc_layer,
                                size=TERM_SIZE,
                                gate_act=TanhActivation(),
                                state_act=TanhActivation(),
                                act=ReluActivation())

lastseq_1_layer = last_seq(input=simple_lstm_layer)

dropout_1_layer = dropout_layer(input=lastseq_1_layer, dropout_rate=0.2)

time_1_output_layer = fc_layer(input=dropout_1_layer, size=4, act=SoftmaxActivation())

label_1 = data_layer(name='label_0', size=4)

time_1_cost = classification_cost(name='<--cost0-->', input=time_1_output_layer, label=label_1)

time_1_value = maxid_layer(input=time_1_output_layer)

cost_output.append(time_1_cost)


output.append(time_1_value)


for i in range(1, TERM_SIZE):

    with mixed_layer(size=TERM_SIZE) as m:
        for layer in output:
            m += layer
    paramAttr = ParameterAttribute(initial_max=1.0, initial_min=-1.0)

    key = 'label_%s' % i

    label = data_layer(name=key, size=4)

    recent_layer = fc_layer(input=m, size=TERM_SIZE, act=ReluActivation())

    fc_nn_layer = fc_layer(input=recent_layer, act=ReluActivation(), size=TERM_SIZE*2)

    dropout_nn_layer = dropout_layer(input=fc_nn_layer, dropout_rate=0.2)

    output_layer = fc_layer(input=dropout_nn_layer, size=4, act=SoftmaxActivation())

    time_output = classification_cost(name='<--cost%s-->' % i, input=output_layer, label=label)

    time_value = maxid_layer(output_layer)

    output.append(time_value.outputs.ids)
    cost_output.append(time_output)

outputs(cost_output)





