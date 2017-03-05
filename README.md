# Traffic Prediction


> ASC 2017 DNN Problem, This is the current method for this problem. I will move on to find a more accurate way to solve this prediction.


## Preprocess

**install Paddle**

## Usage

```
cd data
sh prepare.sh

cd ..

sh train.sh

sh predict.sh

```

## Info

#### preprocess.py

* `points_to_point()` convert the graph.csv to a txt. one center link with other adjacency links.
* `get_points_count_list` generate a list (link, the number of the adjacency links).
* `split_dataset()` split the whole csv file into small txt file.
* `get_predict_data()` generate prediction data

## Network


![](https://github.com/wondervictor/Traffic_Prediction/blob/master/images/network.png)


## Any Problem

**Contact me:** *wonderstruckvictor@hotmail.com*

## Licence

This is under the **Apache 2.0 Licence**
