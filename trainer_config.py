from paddle.trainer_config_helpers import *
import paddle.trainer.config_parser as cp
import numpy as np
import logging
import math

is_predict = get_config_arg('is_predict', bool, False)
nearby_num = get_config_arg('nearby_num', int, 0)
subnode_num = get_config_arg('subnode', int, 0)
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

NODE_NUM = nearby_num + subnode_num + 1

define_py_data_sources2(
    train_list=train,
    test_list=test,
    module="data_provider",
    obj=process,
    args={
        'num': NODE_NUM,
        'point': point,
    }
)

batch_size = 48

if is_predict:
    batch_size = 1

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4)
)

TERM_SIZE = 24

# cost
costs = []

# input
center_data = data_layer(name='data_0', size=TERM_SIZE)
nearby_nodes_inputs = []
nearby_2_nodes_inputs = []
counter = 1
for i in range(nearby_num):
    key = "data_%s" % counter
    nearby_nodes_inputs.append(data_layer(name=key, size=TERM_SIZE))
    counter += 1

for j in range(subnode_num):
    key = "data_%s" % counter
    nearby_2_nodes_inputs.append(data_layer(name=key, size=TERM_SIZE))
    counter += 1






#
# input_fc_1_layer = fc_layer(input=input_data,
#                             size=NODE_NUM,
#                             act=ReluActivation())
#
# bias_attrs_2 = ParameterAttribute(name='bias_attr2',
#                                   learning_rate=1.0,
#                                   initial_mean=0,
#                                   initial_std=0.)
# para_attr_2 = ParameterAttribute(name='para_attr2',
#                                  initial_mean=0.,
#                                  learning_rate=2.0,
#                                  initial_std=0.01/math.sqrt(NODE_NUM*4))
#
# input_fc_2_layer = fc_layer(input=input_fc_1_layer,
#                             size=NODE_NUM*4,
#                             act=ReluActivation(),
#                             param_attr=para_attr_2,
#                             bias_attr=bias_attrs_2)
#
# input_lstm_layer = lstmemory(input=input_fc_2_layer, act=ReluActivation())
#
# input_aggrerate = concat_layer(input=[input_fc_2_layer, input_lstm_layer])
#
# drop_1_layer = dropout_layer(input=input_aggrerate, dropout_rate=0.1)
#
# drop_param = ExtraLayerAttribute(drop_rate=0.1)
#
# bias_attrs_3 = ParameterAttribute(name='bias_attr3', learning_rate=1, initial_mean=0, initial_std=0.1)
#
# para_attr_3 = ParameterAttribute(name='para_attr3',
#                                  initial_mean=0.,
#                                  learning_rate=1,
#                                  initial_std=0.01/math.sqrt(NODE_NUM*NODE_NUM))
#
# fc_2_layer = fc_layer(input=drop_1_layer,
#                       size=NODE_NUM*NODE_NUM,
#                       act=ReluActivation(),
#                       param_attr=para_attr_3,
#                       bias_attr=bias_attrs_3,
#                       layer_attr=drop_param)
#
# lstm_2_layer = simple_lstm(input=fc_2_layer, size=NODE_NUM*NODE_NUM, act=ReluActivation())
#
# input_concat = concat_layer(input=input_data)
# con_layers = concat_layer(input=[input_concat, lstm_2_layer])
#
# labels = []
#
# for i in range(TERM_SIZE):
#     labels.append(data_layer('label_%s' % i, size=4))
#
# SIZE = TERM_SIZE
#
# for i in range(0, TERM_SIZE):
#     bias_attrs_tmp_1 = ParameterAttribute(name='bias_attr_tmp_1_%s' % i,
#                                           learning_rate=1,
#                                           initial_mean=0.,
#                                           initial_std=0.001)
#     para_attr_tmp_1 = ParameterAttribute(name='para_attr_tmp_1_%s' % i,
#                                          initial_mean=0.,
#                                          learning_rate=1,
#                                          initial_std=0.001/math.sqrt(NODE_NUM*4))
#
#     fc_tmp_layer = fc_layer(input=con_layers,
#                             size=NODE_NUM * 4,
#                             act=TanhActivation(),
#                             bias_attr=bias_attrs_tmp_1,
#                             param_attr=para_attr_tmp_1
#                             )
#     con_layers = concat_layer(input=[fc_tmp_layer, input_concat])
#     if i % 2 == 0:
#         lstm_tmp_layer = simple_lstm(input=fc_tmp_layer, size=NODE_NUM*NODE_NUM, act=ReluActivation())
#         con_layers = concat_layer(input=[fc_tmp_layer, lstm_tmp_layer, input_concat])
#     result_aggrerate_layer = last_seq(con_layers)
#     drop_tmp_layer = dropout_layer(input=result_aggrerate_layer, dropout_rate=0.1)
#
#     final_layer = fc_layer(input=drop_tmp_layer,
#                            size=4*NODE_NUM,
#                            act=STanhActivation())
#
#     fc_add_layer = fc_layer(input=concat_layer(input=[final_layer, last_seq(input_concat), drop_tmp_layer]), size=NODE_NUM * 4, act=ReluActivation())
#
#     drop_tmp_2_layer = dropout_layer(input=fc_add_layer, dropout_rate= 0.2)
#
#     time_value = fc_layer(input=drop_tmp_2_layer, size=4, act=SoftmaxActivation())
#
#     if not is_predict:
#         ecost = classification_cost(input=time_value, name='cost%s' % i, label=labels[i])
#         costs.append(ecost)
#     else:
#         value = maxid_layer(time_value)
#         costs.append(value)
# outputs(costs)