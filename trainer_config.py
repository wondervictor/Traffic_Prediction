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

data_layers = []
for i in range(NODE_NUM):
    key = 'data_%s' % i
    data_layers.append(data_layer(name=key, size=TERM_SIZE))


lstm_outputs = []
for i in range(NODE_NUM):
    lstm_output = lstmemory(input=data_layers[i], act=ReluActivation())
    last = pooling_layer(input=lstm_output, pooling_type=AvgPooling(), agg_level=AggregateLevel.EACH_SEQUENCE)
    lstm_outputs.append(last)


label = data_layer(name='label', size=4)
layer_attr = ExtraLayerAttribute(drop_rate=0.5)
param_atte = ParameterAttribute(initial_std=0.0)
fc_1_layer = fc_layer(input=lstm_outputs, size=18, act=ReluActivation(), param_attr=param_atte, layer_attr=layer_attr)
fc_2_layer = fc_layer(input=fc_1_layer, size=9, act=ReluActivation(), param_attr=param_atte)
output_layer = fc_layer(input=fc_2_layer, size=4, act=SoftmaxActivation(), param_attr=param_atte)

cost = classification_cost(input=output_layer, label=label)
