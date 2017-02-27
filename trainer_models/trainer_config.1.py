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
    learning_rate=0.01,
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

###### CONFIGURATION #######

lstm_1_outputs = []
lstm_pool_outputs = []
param_attr = ParameterAttribute()
layer_attr = ExtraLayerAttribute(drop_rate=0.2)

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

input_important_data = inputs_data[1]

rnn = recurrent_layer(input=input_important_data, act=ReluActivation())
lstm_fc_layers = []
rnn_pool = pooling_layer(input=rnn, pooling_type=AvgPooling())

for data in lstm_1_outputs:
    fc_1_layer = fc_layer(input=data, size=TERM_SIZE, act=TanhActivation())
    lstm_fc_layers.append(fc_1_layer)

all_layer_1 = fc_layer(input=lstm_fc_layers, size=TERM_SIZE*NODE_NUM, act=ReluActivation())
all_layer_2 = fc_layer(input=all_layer_1, size=TERM_SIZE*TERM_SIZE, act=ReluActivation(),layer_attr=layer_attr)
all_layer = fc_layer(input=all_layer_2, size=TERM_SIZE, act=ReluActivation())

hidden_4 = fc_layer(input=[rnn_pool, all_layer, hidden3, fc_hidden_pool], size=TERM_SIZE, act=ReluActivation())
dropout_layer_1 = dropout_layer(input=hidden_4, dropout_rate=0.2)

outLayer = fc_layer(input=dropout_layer_1, size=4, act=SoftmaxActivation())

outputs(classification_cost(input=outLayer, label=label))
