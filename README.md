# SSLRec



[User Guide] | [Datasets] | [Models]

[User Guide]: https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md
[Models]: https://github.com/HKUDS/SSLRec/blob/main/docs/Models.md
[Datasets]: https://rxubin.com

**SSLRec** is a PyTorch-based deep learning framework for recommender systems enhanced by self-supervised learning techniques. 
It's user-friendly and contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, and state-of-the-art research models. 
**SSLRec** offers a vast array of utility functions and an easy-to-use interface that simplifies the development and evaluation of recommendation models.

<p align="center">
<img src="sslrec.png" alt="SSLRec" />
</p>

Our library includes 22 self-supervised learning recommendation algorithms, covering five major categories:

+ General Collaborative Filtering
+ Sequential Recommendation
+ Multi-behavior Recommendation
+ Social Recommendation
+ Knowledge-aware Recommendation

We offer a unified training, validation, and testing process for each category, along with a standardized data preprocessing method using publicly available datasets. 
This enables the quick reproduction of various models and the fair comparison of different methods.

## Highlighted Features

+ 🧩**Flexible Modular Architecture.** SSLRec framework features a modular architecture enabling effortless customization and combination of modules to create personalized recommendation models.


+ 🌟**Diverse Recommendation Scenarios.** The SSLRec library is a versatile tool for researchers and practitioners building effective recommendation models across diverse recommender system research lines.


+ 💡**Comprehensive State-of-the-Art Models.** We offer a wide range of SSL-enhanced recommendation models for various scenarios, enabling researchers to evaluate them using advanced techniques, driving innovation in the field.


+ 📊**Unified Data Feeder and Standard Evaluation Protocols.** The SSLRec framework features a unified data feeder and standard evaluation protocols, enabling easy loading and preprocessing of data from various sources and formats, while ensuring objective and fair evaluation of recommendation models.


+ 🛠️**Rich Utility Functions.** The SSLRec library provides a vast array of utility functions that simplify the development and evaluation of recommendation models, incorporating common functionalities of recommender systems and self-supervised learning for graph operations, network architectures, and loss functions.


+ 🤖**Easy-to-Use Interface.** We offer a user-friendly interface that streamlines the training and evaluation of recommendation models, allowing researchers and practitioners to experiment with various models and configurations with ease and efficiency.

## Get Started

SSLRec is implemented under the following development environment:

+ python==3.10.4
+ numpy==1.22.3
+ torch==1.11.0
+ scipy==1.7.3

You can easily train LightGCN using our framework by running the following script:
```
python main.py --model LightGCN
```
This script will run the LightGCN model on the yelp datasets. 

The training configuration for LightGCN is saved in [lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml), 
and you can modify the values to achieve different training effects. Additionally, you can replace LightGCN with other implemented models listed under [Models](./docs/Models.md).

For users who want to learn more, we recommend reading the [User Guide](https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md), which provides detailed explanations of SSLRec concepts and usage, including:
+ SSLRec framework architecture design
+ Implementing your own model in SSLRec
+ Deploying your own datasets in SSLRec
+ Implementing your own training process in SSLRec
+ Automatic hyper-parameter tuning in SSLRec

and so on.

