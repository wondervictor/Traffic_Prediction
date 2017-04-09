from paddle.trainer_config_helpers import *
import paddle.trainer.config_parser as cp
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
    module="data_provider",
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
    learning_rate=0.0001,
    learning_method=MomentumOptimizer(0.0001),
    regularization=L2Regularization(8e-4)
)

TERM_SIZE = 24
NODE_NUM = num


# cost
costs = []

# input
input_data = []
for i in range(NODE_NUM):
    key = "data_%s" % i
    input_data.append(data_layer(name=key, size=TERM_SIZE))


# input_fc
#embeddings = []
# for i in range(NODE_NUM):
#     key = "data_%s" % i
#     embeddings.append(embedding_layer(input=input_data[i], size=TERM_SIZE))
input_fc_1_layer = fc_layer(input=input_data, size=NODE_NUM, act=ReluActivation())

input_fc_2_layer = fc_layer(input=input_fc_1_layer, size=NODE_NUM*4, act=ReluActivation())

input_lstm_layer = lstmemory(input=input_fc_2_layer, act=ReluActivation())

input_aggrerate = concat_layer(input=[input_fc_2_layer, input_lstm_layer])

drop_1_layer = dropout_layer(input=input_aggrerate, dropout_rate=0.1)

fc_2_layer = fc_layer(input=drop_1_layer, size=NODE_NUM*NODE_NUM, act=ReluActivation())

lstm_2_layer = simple_lstm(input=fc_2_layer, size=NODE_NUM*NODE_NUM, act=ReluActivation())
#
# con_layers = concat_layer(input=[fc_2_layer, lstm_2_layer])
#
# input_2_aggrerate = last_seq(input=con_layers)
#
# # one timstamp
#
# first_timestamp_value = fc_layer(input=input_2_aggrerate, size=4, act=SoftmaxActivation())
# cost = classification_cost(input=first_timestamp_value, name='cost0', label=data_layer(name='label_0', size=4))
# costs.append(cost)

input_concat = concat_layer(input=input_data)
con_layers = concat_layer(input=[input_concat, lstm_2_layer])

labels = []

for i in range(TERM_SIZE):
    labels.append(data_layer('label_%s' % i, size=4))

SIZE = TERM_SIZE

for i in range(0, TERM_SIZE):
    fc_tmp_layer = fc_layer(input=con_layers, size=NODE_NUM * 4, act=TanhActivation())
    con_layers = concat_layer(input=[fc_tmp_layer, input_concat])
    if i % 2 == 0:
        lstm_tmp_layer = simple_lstm(input=fc_tmp_layer, size=NODE_NUM*NODE_NUM, act=ReluActivation())
        con_layers = concat_layer(input=[fc_tmp_layer, lstm_tmp_layer, input_concat])
    result_aggrerate_layer = last_seq(con_layers)
    drop_tmp_layer = dropout_layer(input=result_aggrerate_layer, dropout_rate=0.1)
    time_value = fc_layer(input=drop_tmp_layer, size=4, act=SoftmaxActivation())
    ecost = classification_cost(input=time_value, name='cost%s' % i, label=labels[i])
    costs.append(ecost)
outputs(costs)