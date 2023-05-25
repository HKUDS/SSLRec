# SSLRec: A Self-Supervised Learning Library for Recommendation 🚀



[User Guide] | [Datasets] | [Models]

[User Guide]: https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md
[Models]: https://github.com/HKUDS/SSLRec/blob/main/docs/Models.md
[Datasets]: https://github.com/HKUDS/SSLRec/blob/main/docs/Models.md

**SSLRec** is a PyTorch-based deep learning framework for recommender systems enhanced by self-supervised learning techniques. 
It's user-friendly and contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, and state-of-the-art research models. 
**SSLRec** offers a vast array of utility functions and an easy-to-use interface that simplifies the development and evaluation of recommendation models.

<p align="center">
<img src="sslrec.png" alt="SSLRec" />
</p>

Our library includes various self-supervised learning recommendation algorithms, covering five major categories:

+ General Collaborative Filtering
+ Sequential Recommendation
+ Multi-Behavior Recommendation
+ Social Recommendation
+ Knowledge Graph-enhanced Recommendation

We offer a unified training, validation, and testing process for each category, along with a standardized data preprocessing method using publicly available datasets. 
This enables the quick reproduction of various models and the fair comparison of different methods.

## Highlighted Features

+ 🧩**Flexible Modular Architecture.** SSLRec framework features a modular architecture enabling effortless customization and combination of modules to create personalized recommendation models.


+ 🌟**Diverse Recommendation Scenarios.** The SSLRec library is a versatile tool for researchers and practitioners building effective recommendation models across diverse recommender system research lines.


+ 💡**Comprehensive State-of-the-Art Models.** We offer a wide range of SSL-enhanced recommendation models for various scenarios, enabling researchers to evaluate them using advanced techniques, driving innovation in the field.


+ 📊**Unified Data Feeder and Standard Evaluation Protocols.** The SSLRec framework features a unified data feeder and standard evaluation protocols, enabling easy loading and preprocessing of data from various sources and formats, while ensuring objective and fair evaluation of recommendation models.


+ 🛠️**Rich Utility Functions.** The SSLRec library provides a vast array of utility functions that simplify the development and evaluation of recommendation models, incorporating common functionalities of recommender systems and self-supervised learning for graph operations, network architectures, and loss functions.


+ 🤖**Easy-to-Use Interface.** We offer a user-friendly interface that streamlines the training and evaluation of recommendation models, allowing researchers and practitioners to experiment with various models and configurations with ease and efficiency.

## Implemented Models
We are continuously adding new self-supervised models to the SSLRec framework. Stay tuned for updates! 🔍

Here, we list the implemented models in abbrevation. For more detailed information, please refer to [Models](./docs/Models.md).

**General Collaborative Filtering**

+ [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) (SIGIR'20), [SGL](https://arxiv.org/pdf/2010.10783.pdf) (SIGIR'21), [HCCF](https://arxiv.org/pdf/2204.12200.pdf) (SIGIR'22), [SimGCL](https://www.researchgate.net/profile/Junliang-Yu/publication/359788233_Are_Graph_Augmentations_Necessary_Simple_Graph_Contrastive_Learning_for_Recommendation/links/624e802ad726197cfd426f81/Are-Graph-Augmentations-Necessary-Simple-Graph-Contrastive-Learning-for-Recommendation.pdf?ref=https://githubhelp.com) (SIGIR'22), [NCL](https://arxiv.org/pdf/2202.06200.pdf) (WWW'22), [DirectAU](https://dl.acm.org/doi/pdf/10.1145/3534678.3539253) (KDD'22), [LightGCL](https://arxiv.org/pdf/2302.08191.pdf) (ICLR'23)

**Sequential Recommendation**

+ [BERT4Rec](https://arxiv.org/pdf/1904.06690.pdf) (CIKM'19), [CL4SRec](https://arxiv.org/pdf/2010.14395.pdf) (ICDE'22), [DuoRec](https://arxiv.org/pdf/2110.05730.pdf) (WSDM'22), [ICLRec](https://arxiv.org/pdf/2202.02519.pdf) (WWW'22), [DCRec](https://arxiv.org/pdf/2303.11780.pdf) (WWW'23)

**Social Recommendation**

+ [MHCN](https://arxiv.org/pdf/2101.06448.pdf) (WWW'21), [KCGN](https://par.nsf.gov/servlets/purl/10220297) (AAAI'21), [SMIN](https://arxiv.org/pdf/2110.03958.pdf) (CIKM'21), [SDCRec](https://web.archive.org/web/20220712110150id_/https://dl.acm.org/doi/pdf/10.1145/3477495.3531780) (SIGIR'22), [DcRec](https://arxiv.org/pdf/2208.08723.pdf) (CIKM'22)

**Knowledge-aware Recommendation**
+ [KGCL](https://arxiv.org/pdf/2205.00976.pdf) (SIGIR'22)

**Multi-behavior Recommendation**
+ [MBGNN](https://arxiv.org/pdf/2110.03969.pdf) (SIGIR'21), [HMG-CR](https://arxiv.org/pdf/2109.02859.pdf) (ICDM'21), [S-MBRec](http://www.shichuan.org/doc/134.pdf) (IJCAI'22), [CML](https://arxiv.org/pdf/2202.08523.pdf) (WSDM'2022), [KMCLR](https://arxiv.org/pdf/2301.05403.pdf) (WSDM'23), 

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

## Improve Our Framework Together🤝
If you encounter a bug or have any suggestions, please let us know by [filing an issue](https://github.com/HKUDS/SSLRec/issues). 

We welcome all contributions, from bug fixes to new features and extensions. 🙌
