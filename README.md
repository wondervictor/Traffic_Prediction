# Traffic Prediction

**A Deep Learning Prediction Task Supported by PaddlePaddle**


### Introduction

1. Construct a easy Graph model by combine the adjacent nodes together

	* Collect spatio information from `graph.csv`, then construct a Center- MultiOrder-Adjacent Nodes Model (中心-多阶邻接节点图模型)
	* MultiOrder adjacent nodes can lengthen the predicting period

![](https://github.com/wondervictor/Blog/blob/master/Traffic_Graph.png?raw=true)



2. ResLSTM (Residual Long Short Term Memory)
	
	* Inpired by ResNet(which can deepen the neural network to more than 150 layers) ([Deep Residual Network](https://arxiv.org/abs/1512.03385))
	* LSTM can deal with time sequence well
	* Residual can contribute to the loss and reasonable gradient

![](https://github.com/wondervictor/Blog/blob/master/ResLSTM.png?raw=true)

3. Construct the full network

	* Input speed sequences	* Bidirectional LSTM can learn from the former speed and latter speed to deal with the relation of the adjacent nodes and the center node.
	* Use fully connected layers to extract the spatio feature.
	* End-to-End Trainable
	* LSTM and ReLU can lower the possibility of Gradient Vanishing
![](https://github.com/wondervictor/Blog/blob/master/network.png?raw=true)


### Usage

#### Install PaddlePaddle
	
[go to the PaddlePaddle documentation](http://www.paddlepaddle.org/doc/build/build_from_source.html#build-and-install)

##### Prepare data

```
cd data

```

* get point list for training (Graph Constructing)

```
points_to_point()
get_points_count_list_2()
```

* remove zeros
    
```
split_data.split_by_remove_some_timestamps('speeds.csv',[(from_time, to_time)],'speed_nzero.csv')
```

* validation set

```
split_data.split_out(filename, [(from_time, to_time)],[output filenames])
``` 

* test data

```
split_data.get_test_data('test_speeds.csv', 'train_speeds.csv', 'speed_no_valid.csv', [(from_time, to_time)])                                                                                   dataset = create_dataset('test_speeds.csv')
```

* create data set

```
dataset = create_dataset('test_speeds.csv')
get_speed_data_2(dataset, 'test')

dataset = create_dataset('train_speeds.csv')
get_speed_data_2(dataset, 'train')

dataset = create_dataset('speeds_without_zero.csv')
get_predict_data_2(dataset)
dataset = create_dataset('VadiationSet/419_6_10.csv')
get_predict_valid(dataset)
```

##### run

```
sh run.sh
```

### Advantages

* End-to-End trainable
* Maybe high accuracy :)
* Generalization ability and robustness
* make full use of data provided

### Drawbacks

* training speed
* Framework (PaddlePaddle) is too slow and stupid

### Licence

**This project is under the Apache-2.0**


