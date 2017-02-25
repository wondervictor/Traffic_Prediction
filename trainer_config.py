from paddle.trainer_config_helpers import *

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


NODE_NUM = 3
TERM_SIZE = 12

inputs_data = []
for i in range(NODE_NUM):
    key = 'data_%s' % i
    inputs_data.append(data_layer(name=key, size=TERM_SIZE))
label = data_layer(name='label', size=4)

layer_1_outputs = []
for i in range(NODE_NUM):
    hidden_1 = fc_layer(input=inputs_data[i], size=TERM_SIZE, act=SigmoidActivation())
    hidden_2 = fc_layer(input=hidden_1, size=2*TERM_SIZE, act=SigmoidActivation())
    layer_1_outputs.append(hidden_2)

hidden_3 = fc_layer(input=layer_1_outputs, size=NODE_NUM*12, act=SigmoidActivation())
hidden_4 = fc_layer(input=hidden_3, size=TERM_SIZE, act=SigmoidActivation())
output_layer = fc_layer(input=hidden_4, size=4, act=SoftmaxActivation())
cost = classification_cost(input=output_layer, label=label)
outputs(cost)




# NODE_NUM = 3
# TERM_SIZE = 12
#
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

# lstm_1 = lstmemory(input=fc_1_layer, reverse=False, act=ReluActivation())

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

# cost = classification_cost(input=output_layer, label=label)
