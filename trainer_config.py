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
    lstm_outputs.append(lstmemory(input=data_layers[i], act=ReluActivation()))
label = data_layer(name='label', size=1)

layer_attr = ExtraLayerAttribute(drop_rate=0.5)
param_atte = ParameterAttribute(initial_std=0.0,)
fc_1_layer = fc_layer(input=lstm_outputs, size=9, act=ReluActivation(), param_attr=param_atte, layer_attr=layer_attr)
fc_2_layer = fc_layer(input=fc_1_layer, size=4, act=ReluActivation(), param_attr=param_atte)
last = pooling_layer(input=fc_2_layer, pooling_type=AvgPooling())
cost = regression_cost(input=last, label=label)


# data = data_layer(name='data', size=328)
# label = data_layer(name='label', size=328)
#
# fc_1_layer = fc_layer(input=data, size=164, act=ReluActivation())
# fc_2_layer = fc_layer(input=data, size=164, act=ReluActivation())
#
# fc_3_layer = fc_layer(input=[fc_1_layer, fc_2_layer], size = 328, act=ReluActivation())
#
# cost = regression_cost(input=fc_3_layer, label=label)
# outputs(cost)



# fc_1_layer = fc_layer(input=data, size=328, act=ReluActivation())
# layer_param = ExtraAttr(drop_rate=0.6)
# fc_2_layer = fc_layer(input=fc_1_layer, size=328*328, act=ReluActivation(), layer_attr=layer_param)
# out_1_layer = fc_layer(input=fc_2_layer, size=328, act=ReluActivation())
# cost = regression_cost(input=out_1_layer, label=label)
# outputs(cost)