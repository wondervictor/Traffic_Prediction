from paddle.trainer_config_helpers import *
is_predict = get_config_arg('is_predict', bool, False)

define_py_data_sources2(
    train_list='data/train.list',
    test_list='data/test.list',
    module='data_provider',
    obj='process',
    args={
        'num': 3
    }
)

batch_size = 288

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer(),
    regularization=L2Regularization(8e-4)
)


# data = data_layer(name='data', size=12)
# label = data_layer(name='label', size=4)
#
# fc_1_layer = fc_layer(input=data, size=24, act=SigmoidActivation())
# output_layer = fc_layer(input=fc_1_layer, size=4, act=SoftmaxActivation())
# cost = classification_cost(input=output_layer, label=label)
# outputs(cost)


NODE_NUM = 3
TERM_SIZE = 12

inputs_data = []
for i in range(NODE_NUM):
    key = 'data_%s' % i
    inputs_data.append(data_layer(name=key, size=TERM_SIZE))

label = data_layer(name='label', size=4)

lstm_1_outputs = []
lstm_pool_outputs = []
#fc_1_layer_output = []

for data in inputs_data:
    lstm_data_1_layer = lstmemory(input=data, act=ReluActivation())
    lstm_1_outputs.append(lstm_data_1_layer)
    lstm_pool = pooling_layer(input=lstm_data_1_layer, pooling_type=SumPooling())
    lstm_pool_outputs.append(lstm_pool)

fc_hidden1 = fc_layer(input=inputs_data, size=NODE_NUM, act=ReluActivation())
fc_hidden2 = fc_layer(input=fc_hidden1, size=NODE_NUM*NODE_NUM, act=ReluActivation())

sim_lstm = simple_lstm(input=fc_hidden2, size=TERM_SIZE, act=ReluActivation())
fc_hidden_pool = pooling_layer(input=sim_lstm, pooling_type=SumPooling())


hidden1 = fc_layer(input=lstm_pool_outputs, size=NODE_NUM, act=ReluActivation())
hidden2 = fc_layer(input=hidden1, size=NODE_NUM*NODE_NUM, act=ReluActivation())
hidden3 = fc_layer(input=hidden2, size=TERM_SIZE, act=ReluActivation())
lstm_fc_layers = []

for data in lstm_1_outputs:
    fc_1_layer = fc_layer(input=data, size=TERM_SIZE, act=TanhActivation())
    lstm_fc_layers.append(fc_1_layer)

all_layer = fc_layer(input=lstm_fc_layers, size=TERM_SIZE, act=SigmoidActivation())

hidden_4 = fc_layer(input=[all_layer,hidden3,fc_hidden_pool], size=TERM_SIZE, act=ReluActivation())

outLayer = fc_layer(input=hidden_4, size=4, act=SoftmaxActivation())

outputs(classification_cost(input=outLayer, label=label))



#fc_layer_1_output = fc_layer(input=inputs_data, size=NODE_NUM, act=ReluActivation())
#fc_layer_2_output = fc_layer(input=fc_layer_1_output, size=NODE_NUM*NODE_NUM, act=ReluActivation())
#lstm_2_layer = simple_lstm(fc_layer_2_output, size=NODE_NUM,act=ReluActivation(),state_act=SigmoidActivation(), gate_act=SigmoidActivation())

#
# layer_1_outputs = []
# lstm_layers = []



# for i in range(NODE_NUM):
#     name = '%s_lstm' % i
#     lstm_1_layer = lstmemory(name=name, input=inputs_data[i], act=ReluActivation())
#     pool_1_layer = pooling_layer(input=lstm_1_layer, pooling_type=AvgPooling())
#     layer_1_outputs.append(pool_1_layer)
#     #lstm_layers.append(lstm_1_layer)
#
#
# param_attr = ParameterAttribute(initial_std=0.0, initial_mean=1, learning_rate=1e-4)
# layer_attr = ExtraLayerAttribute(drop_rate=0.4)
# hidden_3 = fc_layer(input=layer_1_outputs, size=NODE_NUM*TERM_SIZE, act=ReluActivation(),param_attr=param_attr)
# hidden_4 = fc_layer(input=hidden_3, size=TERM_SIZE*TERM_SIZE, act=ReluActivation(),layer_attr=layer_attr)
# hidden_5 = fc_layer(input=hidden_4, size=TERM_SIZE, act=TanhActivation())

# lstm_fc_2_layer = fc_layer(input=lstm_layers, size=TERM_SIZE, act=TanhActivation(),layer_attr=layer_attr)
# lstm_fc_3_layer = fc_layer(input=lstm_fc_2_layer, size=TERM_SIZE*NODE_NUM, act=ReluActivation(), layer_attr=layer_attr)
# lstm_fc_4_layer = fc_layer(input=lstm_fc_3_layer, size=TERM_SIZE, act=ReluActivation())
# lstm_pool_layer = pooling_layer(input=lstm_fc_4_layer, pooling_type=AvgPooling())

# new_lstms = []
# for i in lstm_layers:
#

#fc_all_layers = fc_layer(input=hidden_5, size=NODE_NUM, layer_attr=layer_attr, act=ReluActivation())
# output_layer = fc_layer(input=hidden_5, size=4, act=SoftmaxActivation())
# cost = classification_cost(input=output_layer, label=label)
#
#

#
# NODE_NUM = 3
# TERM_SIZE = 12

# label = data_layer(name='label', size=4)
#
# data = data_layer(name='data', size=TERM_SIZE)
#
# fc_1_layer = fc_layer(input=data, size=TERM_SIZE, act=LinearActivation())
#
# fc_2_layer = fc_layer(input=fc_1_layer, size=24, act=ReluActivation())
#
# output = fc_layer(input=fc_2_layer, size=4, act=SoftmaxActivation())
#
# cost = classification_cost(input=output, label=label)
#
# outputs(cost)
#
# lstm_1 = lstmemory(input=fc_1_layer, reverse=False, act=ReluActivation())
#
# last = pooling_layer(input=lstm_1, pooling_type=MaxPooling())
# label = data_layer(name='label', size=1)
# cost = regression_cost(input=last, label=label)
# outputs(cost)

# data_layers = []
# for i in range(NODE_NUM):
#     key = 'data_%s' % i
#     data_layers.append(data_layer(name=key, size=TERM_SIZE))
#
#
# lstm_outputs = []
# for i in range(NODE_NUM):
#     lstm_output = simple_lstm(input=data_layers[i], size=1, act=ReluActivation())
#     last = pooling_layer(input=lstm_output, pooling_type=AvgPooling(), agg_level=AggregateLevel.EACH_SEQUENCE)
#     lstm_outputs.append(last)
#
#
# label = data_layer(name='label', size=4)
# layer_attr = ExtraLayerAttribute(drop_rate=0.5)
# param_atte = ParameterAttribute(initial_std=0.0)
# fc_1_layer = fc_layer(input=lstm_outputs, size=18, act=ReluActivation(), param_attr=param_atte, layer_attr=layer_attr)
# fc_2_layer = fc_layer(input=fc_1_layer, size=9, act=ReluActivation(), param_attr=param_atte)
# output_layer = fc_layer(input=fc_2_layer, size=4, act=SoftmaxActivation(), param_attr=param_atte)
#
# cost = classification_cost(input=output_layer, label=label)
