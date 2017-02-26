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
