from paddle.trainer_config_helpers import *
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


def forward_network(index, input_layer, output, size):
    paramAttr = ParameterAttribute(name='common', initial_max=1.0, initial_min=-1.0)
    key = 'label_%s' % index
    label = data_layer(name=key, size=4)
    recent_layer = fc_layer(input=output, size=size, act=ReluActivation())
    fc_nn_layer = fc_layer(input=[recent_layer, output], act=ReluActivation, size=size*2, paramAttr=paramAttr)
    dropout_1_layer = dropout_layer(input=fc_nn_layer, dropout_rate=0.2)
    output_layer = fc_layer(input=dropout_1_layer, size=4, act=SoftmaxActivation())
    time_output = classification_cost(input=output_layer, label=label)
    time_value = maxid_layer(output_layer)
    output.append(time_value)
    return time_output, output



TERM_SIZE = 24
NODE_NUM = num

# input
input_data = []
for i in range(NODE_NUM):
    key = "data_%s" % i
    input_data.append(data_layer(name=key, size=TERM_SIZE))

output = []
cost_output = []

paramAttr = ParameterAttribute(name='common', initial_max=1.0, initial_min=-1.0)


# 1 lstmemory cells
lstm_cells = []

for data in input_data:
    lstm_cell = lstmemory(input=data, gate_act=TanhActivation(), act=ReluActivation(), state_act=TanhActivation(),size=1)
    lstm_cells.append(lstm_cell)

lstm_fc_layer = fc_layer(input=lstm_cells, size=3, act=SigmoidActivation(),param_attr=paramAttr)

simple_lstm_layer = simple_lstm(input=lstm_fc_layer, size=TERM_SIZE,paramAttr=paramAttr, gate_act=TanhActivation(), state_act=TanhActivation(),
                                act=SigmoidActivation())
lastseq_1_layer = last_seq(input=simple_lstm_layer)

dropout_1_layer = dropout_layer(input=last_seq, dropout_rate=0.2)

time_1_output_layer = fc_layer(input=dropout_1_layer, size=4, act=SoftmaxActivation(), paramAttr=paramAttr)

label_1 = data_layer(name='label_1', size=4)

time_1_output = classification_cost(input=time_1_output_layer, label=label_1)

time_1_value = maxid_layer(input=time_1_output_layer)

cost_output.append(time_1_output)
output.append(time_1_value)

for i in range(1, TERM_SIZE):
    time_output, output_arr = forward_network(i, last_seq, output, TERM_SIZE)
    cost_output.append(time_output)
    output = output_arr

outputs(cost_output)










