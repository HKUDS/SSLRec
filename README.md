# SSLRec

**SSLRec** is an easy-to-use PyTorch-based deep learning framework for recommender systems enhanced by self-supervised learning techniques.
It contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, 
and state-of-the-art research models.

Our library includes 21 self-supervised learning recommendation algorithms, covering four major categories:

+ General Collaborative Filtering
+ Sequential Recommendation
+ Multi-behavior Recommendation
+ Social Recommendation

We provide a unified training, verification and testing process for each category, 
as well as a unified data preprocessing process with public datasets, 
so as to quickly reproduce different models and compare different methods fairly.

## Highlighted Features


## Get Started

SSLRec is implemented under the following development environment:

+ python==3.10.4
+ numpy==1.22.3
+ torch==1.11.0
+ scipy==1.7.3

You can easily use the following script to train LightGCN using our framework:
```
python main.py --model LightGCN
```
This script will run the LightGCN model on the yelp datasets. 

The training configuration of the LightGCN model is stored in [lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml), 
and you can modify the values in it to obtain different training effects. You can also replace LightGCN with other implemented models listed in [Models]().

For acquainted users who wish to learn more, read the [User Guide](https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md), which explains the concepts and usage of SSLRec in much more details, including:
+ Architecture design of SSLRec framework
+ How to implement your own model using SSLRec
+ How to deploy your own datasets in SSLRec
+ How to implement your own training process in SSLRec
+ How to tune hyper-parameters automatically in SSLRec

and so on.

