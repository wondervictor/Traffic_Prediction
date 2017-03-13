from paddle.trainer_config_helpers import *

is_predict = get_config_arg('is_predict', bool, False)
num = get_config_arg('num', int, 0)
point = get_config_arg('point', int, 0)


with open('data/train.list', 'w') as f:
    f.write('data/speed_data/%s.txt' % point)
with open('data/test.list', 'w') as f:
    f.write('data/speed_data/%s.txt' % point)
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

define_py_data_sources2(
    train_list=train,
    test_list=test,
    module='data_provider',
    obj=process,
    args={
        'num': num,
        'point': point,
    }
)

batch_size = 288

if is_predict:
    batch_size = 1

settings(
    batch_size=batch_size,
    learning_rate=0.001,
    learning_method=RMSPropOptimizer(),
    regularization=L2Regularization(8e-4)
)


TERM_SIZE = 24
NODE_NUM = num

# input
input_data = []
for i in range(NODE_NUM):
    key = "data_%s" % i
    input_data.append(data_layer(name=key, size=TERM_SIZE))

output = []

for i in range(TERM_SIZE):

    # 0 - LSTM for one point
    center_data = input_data[0]

    # 0 - LSTM

    lstm_0_layer = lstmemory(input=center_data, act=ReluActivation())

    lstm_0_last_pool = last_seq(input=lstm_0_layer)

    # 1 - LSTM
    lstm_1_layer_outputs = []
    for data in input_data:
        lstm_layer = lstmemory(input=data, act=ReluActivation())
        lstm_1_layer_outputs.append(lstm_layer)

    # 1 - pool - last
    lstm_1_last_pools = []
    for out in lstm_1_layer_outputs:
        last = last_seq(input=out)
        lstm_1_last_pools.append(last)

    # 1 - lstm - output
    lstm_1_outputs = []  # [lstm_1_last_pools, lstm_1_avg_pools]
    lstm_1_outputs.extend(lstm_1_last_pools)
    # lstm_1_outputs.extend(lstm_1_avg_pools)


    # 1 - fc
    fc_1_1_layer = fc_layer(input=lstm_1_outputs, size=TERM_SIZE, act=ReluActivation())
    fc_1_2_layer = fc_layer(input=fc_1_1_layer, size=TERM_SIZE * NODE_NUM, act=ReluActivation())

    # 1 all layers output

    all_1_layers = []  # [lstm_1_last_pools, fc_1_2_layer]
    # all_1_layers.extend(lstm_1_avg_pools)
    all_1_layers.append(fc_1_2_layer)

    # 2 - fc
    fc_2_1_layer = fc_layer(input=input_data, size=TERM_SIZE, act=ReluActivation())
    fc_2_2_layer = fc_layer(input=fc_2_1_layer, size=TERM_SIZE * NODE_NUM, act=ReluActivation())

    # 2 - simple lstm
    simple_2_1_lstm = simple_lstm(input=fc_2_2_layer, size=TERM_SIZE * NODE_NUM, act=ReluActivation())

    # 2 - fc
    fc_2_3_layer = fc_layer(input=simple_2_1_lstm, size=TERM_SIZE, act=ReluActivation())

    # 2 - simple lstm
    simple_2_2_lstm = simple_lstm(input=fc_2_3_layer, size=TERM_SIZE, act=ReluActivation())

    # 2 - last pool
    last_2_pool = last_seq(input=simple_2_2_lstm)

    # all 2 layers output
    all_2_layers = last_2_pool

    all_ouputs = []
    all_ouputs.extend(all_1_layers)
    all_ouputs.append(all_2_layers)
    all_ouputs.extend(lstm_1_last_pools)
    all_ouputs.append(lstm_0_last_pool)

    all_fc_1_layer = fc_layer(input=all_ouputs, size=TERM_SIZE, act=ReluActivation())
    output_layer = fc_layer(input=all_fc_1_layer, size=4, act=SoftmaxActivation())

    if is_predict:
        maxid = maxid_layer(output_layer)
        output.append(maxid)
    else:
        # label
        label = data_layer(name='label_%s' % i, size=4)
        cost = classification_cost(name='<---- cost %s -- %s--->' % (point, (i + 1) * 5), input=output_layer,
                                   label=label)
        output.append(cost)
outputs(output)
