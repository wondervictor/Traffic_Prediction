from paddle.trainer_config_helpers import *


define_py_data_sources2(
    train_list='data/train.list',
    test_list='data/test.list',
    module='data_provider',
    obj='process'
)

DATA_SIZE = 328
batch_size = 24
settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer,
    regularization=L2Regularization
)


node = data_layer(name='node', size=DATA_SIZE)
label = data_layer(name='label', size=DATA_SIZE)

input_fc = fc_layer(input=node, size=DATA_SIZE, act=ReluActivation())
first_rnn = recurrent_layer(input=input_fc, act=SoftmaxActivation())
hidden_fc = fc_layer(input=first_rnn, size=DATA_SIZE*DATA_SIZE, act=ReluActivation())
second_rnn = recurrent_layer(input=hidden_fc, size=DATA_SIZE*DATA_SIZE, act=SoftmaxActivation())
output_fc = fc_layer(input=second_rnn, size=DATA_SIZE, act=SoftmaxActivation())

cost = classification_cost(input=output_fc, label=label)
outputs(cost)

