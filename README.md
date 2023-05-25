# SSLRec

<p align="center">
<img src="sslrec.png" alt="SSLRec" />
</p>

**SSLRec** is an easy-to-use PyTorch-based deep learning framework for recommender systems enhanced by self-supervised learning techniques.
It contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, 
and state-of-the-art research models. **SSLRec** offers a rich collection of utility functions and an easy-to-use interface to simplify the process of evaluating and developing recommendation models.

Our library includes 22 self-supervised learning recommendation algorithms, covering five major categories:

+ General Collaborative Filtering
+ Sequential Recommendation
+ Multi-behavior Recommendation
+ Social Recommendation
+ Knowledge-aware Recommendation

We provide a unified training, verification and testing process for each category, 
as well as a unified data preprocessing process with public datasets, 
so as to quickly reproduce different models and compare different methods fairly.

## Highlighted Features

+ 🧩**Flexible Modular Architecture.** SSLRec framework features a modular architecture enabling effortless customization and combination of modules to create personalized recommendation models.


+ 🌟**Diverse Recommendation Scenarios.** The SSLRec library is a versatile tool for researchers and practitioners building effective recommendation models across diverse recommender system research lines.


+ 💡**Comprehensive State-of-the-Art Models** We offer a wide range of SSL-enhanced recommendation models for various scenarios, enabling researchers to evaluate them using advanced techniques, driving innovation in the field.


+ 📊**Unified Data Feeder and Standard Evaluation Protocols** The SSLRec framework features a unified data feeder and standard evaluation protocols, enabling easy loading and preprocessing of data from various sources and formats, while ensuring objective and fair evaluation of recommendation models.


+ 🛠️**Rich Utility Functions** The SSLRec library provides a vast array of utility functions that simplify the development and evaluation of recommendation models, incorporating common functionalities of recommender systems and self-supervised learning for graph operations, network architectures, and loss functions.


+ 🤖**Easy-to-Use Interface** We offer a user-friendly interface that streamlines the training and evaluation of recommendation models, allowing researchers and practitioners to experiment with various models and configurations with ease and efficiency.

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
and you can modify the values in it to obtain different training effects. You can also replace LightGCN with other implemented models listed in [Models](./docs/Models.md).

For acquainted users who wish to learn more, read the [User Guide](https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md), which explains the concepts and usage of SSLRec in much more details, including:
+ Architecture design of SSLRec framework
+ How to implement your own model using SSLRec
+ How to deploy your own datasets in SSLRec
+ How to implement your own training process in SSLRec
+ How to tune hyper-parameters automatically in SSLRec

and so on.

