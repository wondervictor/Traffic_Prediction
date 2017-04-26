from paddle.trainer_config_helpers import *
import paddle.trainer.config_parser as cp
import numpy as np
import logging
import math

# 21499 0.288675134595
# 21499 0.288675134595

#27500 1.20761472885
#27500 1.20761472885
#27500 1.48604620834

#23683 1.39940463531

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

bias_param = ParameterAttribute(l2_rate=0., initial_std=0.0001, initial_mean=0.)
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
backward_first_lstm_layer = lstmemory(input=first_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param, reverse=True)

forward_second_lstm_layer = lstmemory(input=second_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param)
backward_second_lstm_layer = lstmemory(input=second_order_region_fc_layer,act=ReluActivation(), bias_attr=bias_param, reverse=True)

first_order_region_fc_2_layer = fc_layer(input=[forward_first_lstm_layer, backward_first_lstm_layer], size=region_1_node_num, act=ReluActivation(),bias_attr=bias_param)
second_order_region_fc_2_layer = fc_layer(input=[forward_second_lstm_layer, backward_second_lstm_layer], size=region_2_node_num, act=ReluActivation(),bias_attr=bias_param)


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

    res_fc = fc_layer(input=res, size=NODE_NUM * 4, act=ReluActivation(), bias_attr=bias_param)
    if i % 2 == 0:
        add_res_layer = lstmemory(input=res_fc,act=ReluActivation(),bias_attr=bias_param)
        # add_res_layer = simple_lstm(input=res, size=NODE_NUM, act=ReluActivation(), bias_param_attr=bias_param)
        # add_res_layer = addto_layer(input=[res_lstm, res], act=ReluActivation(), bias_attr=bias_param)
    else:
        # res_fc = fc_layer(input=res, size=NODE_NUM*4, act=ReluActivation(), bias_attr=bias_param)
        res_fc_1 = fc_layer(input=res_fc, size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)
        add_res_layer = fc_layer(input=[res, res_fc_1], size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)

    add_nearby_res = nearby_res

    if i % 2 != 0:
        nearby_res_fc = fc_layer(input=res, size=NODE_NUM*4, act=ReluActivation(), bias_attr=bias_param)

    # if i % 6 == 0:
    #     add_nearby_res = lstmemory(input=nearby_res_fc, act=ReluActivation(), bias_attr=bias_param)
    #     # add_nearby_res = simple_lstm(input=nearby_res, size=NODE_NUM, act=ReluActivation(), bias_param_attr=bias_param)
    # else:
        # nearby_res_fc = fc_layer(input=res, size=NODE_NUM*4, act=ReluActivation(), bias_attr=bias_param)
        nearby_res_fc_1 = fc_layer(input=nearby_res, size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)
        add_nearby_res = fc_layer(input=[nearby_res, nearby_res_fc_1], size=NODE_NUM, act=ReluActivation(), bias_attr=bias_param)

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


# 0.288675134595

# #.57735026919
# from paddle.trainer_config_helpers import *
# import paddle.trainer.config_parser as cp
# import numpy as np
# import logging
# import math
#
# is_predict = get_config_arg('is_predict', bool, False)
# nearby_num = get_config_arg('nearby_num', int, 0)
# subnode_num = get_config_arg('subnode', int, 0)
# point = get_config_arg('point', int, 0)
#
#
# with open('data/train.list', 'w') as f:
#     f.write('data/train/%s.txt' % point)
# with open('data/test.list', 'w') as f:
#     f.write('data/test/%s.txt' % point)
# process = 'process'
# if is_predict:
#     process = 'process_predict'
#     with open('data/pred.list', 'w') as f:
#         f.write('data/predict_data/%s.txt' % point)
#
# test = 'data/test.list'
# train = 'data/train.list'
# if is_predict:
#     train = None
#     test = 'data/pred.list'
#
# NODE_NUM = nearby_num + subnode_num + 1
#
# define_py_data_sources2(
#     train_list=train,
#     test_list=test,
#     module="data_provider",
#     obj=process,
#     args={
#         'num': NODE_NUM,
#         'point': point,
#     }
# )
#
# batch_size = 12
#
# if is_predict:
#     batch_size = 1
#
# settings(
#     batch_size=batch_size,
#     learning_rate=0.0001,
#     learning_method=RMSPropOptimizer(1e-5,0.9),#MomentumOptimizer(1e-4),#RMSPropOptimizer(epsilon=0.0001,rho=0.95),
#     regularization=L2Regularization(5e-4),
#     gradient_clipping_threshold=25
# )
#
# TERM_SIZE = 24
#
# # cost
# costs = []
#
# # input
# center_data = data_layer(name='data_0', size=TERM_SIZE)
# nearby_nodes_inputs = []
# nearby_2_nodes_inputs = []
# counter = 1
# for i in range(nearby_num):
#     key = "data_%s" % counter
#     nearby_nodes_inputs.append(data_layer(name=key, size=TERM_SIZE))
#     counter += 1
#
# for j in range(subnode_num):
#     key = "data_%s" % counter
#     nearby_2_nodes_inputs.append(data_layer(name=key, size=TERM_SIZE))
#     counter += 1
#
#
# nearby_nodes = []
#
# bias_param = ParameterAttribute(initial_std=0.001/(math.sqrt(NODE_NUM)))
#
# drop_param = ExtraLayerAttribute(drop_rate=0.2)
# # aggregate all nearby nodes
# nearby_fc_layer_1 = fc_layer(input=nearby_nodes_inputs, size=nearby_num, act=ReluActivation(),layer_attr=drop_param,bias_attr=bias_param)
# # all subnodes
# nearby_second_fc_layer = fc_layer(input=nearby_2_nodes_inputs, size=subnode_num, act=ReluActivation(),layer_attr=drop_param,bias_attr=bias_param)
#
# lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=1.)
# bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
# layer_attr = ExtraLayerAttribute(drop_rate=0.4)
#
# lstm_forward_nearby_layer = simple_lstm(input=nearby_fc_layer_1, size=nearby_num, act=ReluActivation(),bias_param_attr=bias_attr, inner_param_attr=lstm_para_attr,mixed_layer_attr=layer_attr)
# lstm_backward_nearby_layer = simple_lstm(input=nearby_fc_layer_1, size=nearby_num, act=ReluActivation(), reverse=True,bias_param_attr=bias_attr)
#
# lstm_forward_second_layer = simple_lstm(input=nearby_second_fc_layer, size=subnode_num, act=ReluActivation(),bias_param_attr=bias_attr, inner_param_attr=lstm_para_attr,mixed_layer_attr=layer_attr)
# lstm_backward_second_layer = simple_lstm(input=nearby_second_fc_layer, size=subnode_num, act=ReluActivation(), reverse=True,bias_param_attr=bias_attr)
#
# nearby_fc_layer_2 = fc_layer(input=[lstm_forward_nearby_layer, lstm_backward_nearby_layer], size=nearby_num*4, act=ReluActivation(), bias_attr=bias_param)
#
# nearby_second_fc_layer_2 = fc_layer(input=[lstm_forward_second_layer, lstm_backward_second_layer], size=subnode_num*4, act=ReluActivation(), bias_attr=bias_param)
#
#
# # nearby_lstm = simple_lstm(input=nearby_fc_layer_2, size=nearby_num,act=ReluActivation())
# #
# # nearby_second_lstm = simple_lstm(input=nearby_second_fc_layer_2, size=subnode_num, act=ReluActivation())
#
# large_drop = ExtraLayerAttribute(drop_rate=0.4)
#
# nearby_all_fc_layer = fc_layer(input=concat_layer(input=[nearby_fc_layer_2, nearby_second_fc_layer_2]),
#                                size=nearby_num*subnode_num,
#                                act=TanhActivation(),
#                                layer_attr=large_drop,
#                                bias_attr=bias_param)
#
# nearby_bias = ParameterAttribute(learning_rate=1.5, initial_mean=0, initial_std=0.001)
#
# nearby_all_aggregate_layer = fc_layer(input=nearby_all_fc_layer,
#                                       size=nearby_num,
#                                       act=ReluActivation(),
#                                       bias_attr=nearby_bias)
#
# # nearby_all_lstm = simple_lstm(input=nearby_all_aggregate_layer, size=nearby_num, act=ReluActivation())
# #
# # center_lstm = simple_lstm(input=center_data, size=1, act=ReluActivation())
#
# center_bias = ParameterAttribute(name='center_nearby_i',learning_rate=2., initial_mean=0., initial_std=0.001)
#
# center_with_nearby_layer = fc_layer(input=[nearby_all_aggregate_layer, center_data, nearby_fc_layer_2, nearby_second_fc_layer_2],
#                                     size=NODE_NUM * 4,
#                                     act=ReluActivation(),
#                                     bias_attr=center_bias
#                                     )
#
#  #
# # center_forward_lstm = simple_lstm(input=center_data, size=1, act=ReluActivation())
# # center_backward_lstm = simple_lstm(input=center_data, size=1, act=ReluActivation(), reverse=True)
# #
# # center_datas = fc_layer(input=[center_backward_lstm, center_forward_lstm], size=1, act=ReluActivation())
# con_layers = concat_layer(input=[center_data, center_with_nearby_layer])
#
# con_layers = fc_layer(input=con_layers, size=NODE_NUM*4)
# increment_bias = ParameterAttribute(name='increment_bias',
#                                     momentum=0.0001,
#                                     l2_rate=0.,
#                                     initial_std=0.001,
#                                     initial_mean=0.)
# increment_layer = fc_layer(input=[center_data, center_with_nearby_layer, nearby_all_aggregate_layer],
#                            size=NODE_NUM*4,
#                            act=TanhActivation(),
#                            bias_attr=increment_bias)
#
# # output_result = []
#
# labels = []
#
# for i in range(TERM_SIZE):
#     labels.append(data_layer('label_%s' % i, size=4))
#
#
# SIZE = TERM_SIZE
# # forward_layer = con_layers
#
# for i in range(0, TERM_SIZE):
#     bias_attrs_tmp_1 = ParameterAttribute(name='bias_attr_tmp_1_%s' % i,
#                                           learning_rate=1.,
#                                           initial_mean=0.,
#                                           l2_rate=0.,
#                                           initial_std=0.001)
#     # para_attr_tmp_1 = ParameterAttribute(name='para_attr_tmp_1_%s' % i,
#     #                                      initial_mean=0.,
#     #                                      learning_rate=2.,
#     #                                      initial_std=0.001/math.sqrt(NODE_NUM*4))
#
#     fc_tmp_layer = fc_layer(input=con_layers,
#                             size=NODE_NUM * 4,
#                             act=ReluActivation(),
#                             bias_attr=bias_attrs_tmp_1
#                             )
#
#     if i % 2 == 0:
#         res_layer = addto_layer(input=[con_layers, increment_layer], act=ReluActivation())
#         drop_tmp_layer = dropout_layer(input=res_layer, dropout_rate=0.4)
#         lstm_tmp_layer = simple_lstm(input=drop_tmp_layer, size=NODE_NUM*4, act=ReluActivation(),bias_param_attr=bias_attr, inner_param_attr=lstm_para_attr,mixed_layer_attr=layer_attr)
#         # lstm_tmp_layer_backward = simple_lstm(input=drop_tmp_layer, size=NODE_NUM, act=ReluActivation(), reverse=True)
#         con_layers = concat_layer(input=[fc_tmp_layer, lstm_tmp_layer, center_with_nearby_layer])
#     else:
#         # con_layers = concat_layer(input=fc_tmp_layer])
#         con_layers = dropout_layer(input=fc_tmp_layer, dropout_rate=0.4)
#
#     con_layers = fc_layer(input=con_layers, size=NODE_NUM*4, act=ReluActivation())
#     result_aggrerate_layer = last_seq(con_layers)
#
#     final_bias = ParameterAttribute(name='final_bias_%s' % i,
#                                     momentum=0.0001,
#                                     l2_rate=0.,
#                                     initial_std=0.001,
#                                     initial_mean=0.)
#     # para_attr = ParameterAttribute(name='final_param_%s' % i,
#     #                                momentum=0.0001,
#     #                                l2_rate=0.,
#     #                                learning_rate=0.0001,
#     #                                initial_std=0.0001/(4*NODE_NUM),
#     #                                initial_mean=0.)
#
#     final_layer = fc_layer(input=result_aggrerate_layer,
#                            size=4*NODE_NUM,
#                            act=STanhActivation(),bias_attr=final_bias)
#
#     # drop_tmp_2_layer = dropout_layer(input=final_layer, dropout_rate=0.3)
#     # forward_layer = drop_tmp_2_layer
#     time_value = fc_layer(input=final_layer, size=4, act=SoftmaxActivation())
#
#     if not is_predict:
#         ecost = classification_cost(input=time_value, name='cost%s' % i, label=labels[i])
#         costs.append(ecost)
#     else:
#         value = maxid_layer(time_value)
#         costs.append(value)
# outputs(costs)
#
#
#
#
#
#
#
# #
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