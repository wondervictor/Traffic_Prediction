from paddle.trainer_config_helpers import *


define_py_data_sources2(
    train_list='data/train.list',
    test_list='data/test.list',
    module='data_provider',
    obj='process'
)


batch_size = 32
settings(
    batch_size=32,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer,
    regularization=L2Regularization
)


data = data_layer(name='data', )





