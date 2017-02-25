from paddle.trainer_config_helpers import *

define_py_data_sources2(
    train_list='data/train.list',
    test_list='data/test.list',
    module='data_provider',
    obj='process'
)

batch_size = 288

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer(),
    regularization=L2Regularization(8e-4)
)

data = data_layer(name='data', size=328)
label = data_layer(name='label', size=328)
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