from paddle.trainer_config_helpers import *
import paddle.trainer.config_parser as cp
import numpy as np
import logging
import math


is_predict = get_config_arg('is_predict', bool, False)
region_1_node_num = get_config_arg('nearby_num', int, 0)
region_2_node_num = get_config_arg('subnode', int, 0)
point = get_config_arg('point', int, 0)


with open('data/train.list', 'w') as f:
    f.write('data/train/%s.txt' % point)
with open('data/test.list', 'w') as f:
    f.write('data/test/%s.txt' % point)
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

NODE_NUM = region_1_node_num + region_2_node_num + 1

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

batch_size = 6

if is_predict:
    batch_size = 1

settings(
    batch_size=batch_size,
    learning_rate=0.0001,
    learning_method=RMSPropOptimizer(),#MomentumOptimizer(1e-4),#RMSPropOptimizer(epsilon=0.0001,rho=0.95),
    regularization=L2Regularization(5e-4),
    gradient_clipping_threshold=25
)

TERM_SIZE = 24

# cost
costs = []

# input
center_data = data_layer(name='data_0', size=TERM_SIZE)
first_region_nodes = []
second_region_nodes = []
counter = 1
for i in range(region_1_node_num):
    key = "data_%s" % counter
    first_region_nodes.append(data_layer(name=key, size=TERM_SIZE))
    counter += 1

for j in range(region_2_node_num):
    key = "data_%s" % counter
    second_region_nodes.append(data_layer(name=key, size=TERM_SIZE))
    counter += 1

bias_param = ParameterAttribute(l2_rate=0., initial_std=0.001, initial_mean=0.)
drop_param = ExtraLayerAttribute(drop_rate=0.3)

first_order_region_fc_layer = fc_layer(input=first_region_nodes,
                                       size=region_1_node_num*4,
                                       act=TanhActivation(),
                                       bias_attr=bias_param,
                                       layer_attr=drop_param
                                       )

second_order_region_fc_layer = fc_layer(input=second_region_nodes,
                                        size=region_2_node_num * 4,
                                        act=TanhActivation(),
                                        bias_attr=bias_param,
                                        layer_attr=drop_param
                                        )

forward_first_lstm_layer = lstmemory(input=first_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param)
# backward_first_lstm_layer = lstmemory(input=first_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param, reverse=True)

forward_second_lstm_layer = lstmemory(input=second_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param)
# backward_second_lstm_layer = lstmemory(input=second_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param, reverse=True)

first_order_region_fc_2_layer = fc_layer(input=forward_first_lstm_layer, size=region_1_node_num, act=TanhActivation(),bias_attr=bias_param)
second_order_region_fc_2_layer = fc_layer(input=forward_second_lstm_layer, size=region_2_node_num, act=TanhActivation(),bias_attr=bias_param)


near_regions_fc_layer = fc_layer(input=[first_order_region_fc_2_layer, second_order_region_fc_2_layer],
                                 size=(region_2_node_num + region_1_node_num)*2,
                                 act=ReluActivation(),
                                 bias_attr=bias_param,
                                 layer_attr=drop_param)

center_concat = concat_layer(input=[center_data,
                                    near_regions_fc_layer]
                             )

nearby_res = fc_layer(input=near_regions_fc_layer, size=NODE_NUM, act=ReluActivation())
res = fc_layer(input=center_concat, size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)
res = fc_layer(input=[res, center_data], size=NODE_NUM, act=TanhActivation())

output_cost = []
labels = []

for i in range(TERM_SIZE):
    labels.append(data_layer('label_%s' % i, size=4))

for i in range(TERM_SIZE):

    final_bias = ParameterAttribute(momentum=0.0001,
                                    l2_rate=0.,
                                    initial_std=0.001,
                                    initial_mean=0.)

    add_res_layer = res

    if i % 2 == 0:
        add_res_layer = simple_lstm(name='add_res_lstm_%s_layer' % i,input=res, size=NODE_NUM, act=ReluActivation(), bias_param_attr=bias_param)
        add_res_layer = addto_layer(input=[add_res_layer, res], act=TanhActivation(), bias_attr=bias_param)
    else:
        res_fc = fc_layer(name='add_res_%s_fc_layer' % i, input=res, size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)
        add_res_layer = addto_layer(input=[res, res_fc], act=TanhActivation(), bias_attr=bias_param)

    add_nearby_res = nearby_res

    if i % 2 != 0:
        nearby_res_fc = fc_layer(input=nearby_res, size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)
        add_nearby_res = addto_layer(input=[nearby_res, nearby_res_fc], act=TanhActivation(), bias_attr=bias_param)

    final_layer = fc_layer(input=[add_res_layer, add_nearby_res],
                           size=NODE_NUM * 4,
                           act=STanhActivation(),
                           bias_attr=final_bias,
                           layer_attr=drop_param
                           )

    res = fc_layer(input=final_layer, size=NODE_NUM, act=TanhActivation(), bias_attr=final_bias)
    nearby_res = add_nearby_res
    time_value = fc_layer(input=last_seq(res), size=4, act=SoftmaxActivation())

    if not is_predict:
        ecost = classification_cost(input=time_value, name='cost%s' % i, label=labels[i])
        output_cost.append(ecost)
    else:
        value = maxid_layer(time_value)
        output_cost.append(value)
outputs(output_cost)