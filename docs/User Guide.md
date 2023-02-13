# User Guide
The user guide contains the following content, you can quickly jump to the corresponding section

+ Architecture Design of SSLRec

## Architecture Design of SSLRec
SSLRec is a unified self-supervised recommendation algorithm framework, 
which includes the following 6 main basic classes.
### DataHandler
**DataHandler** is used to read the raw data, perform data preprocessing (such as converting to a sparse matrix format), and finally organize the data into a DataLoader for training and testing.
In our design, it contains two important functions:
+ ```__init__()```: It stores the original path of the corresponding dataset according to the configuration provided by the user.
+ ```load_data()```: It reads the raw data, performs necessary data preprocessing and finally instantiates ```train_dataloader``` and ```test_dataloader```

We designed different DataHandlers for four categories (i.e., General Collaborative Filtering, Sequential Recommendation, Multi-behavior Recommendation, 
Social Recommendation) respectively. You can get a more detailed understanding by reading the source code of [DataHandlerGeneralCF](https://github.com/HKUDS/SSLRec/blob/main/data_utils/data_handler_general_cf.py).

### Dataset
**Dataset** inherits the ```torch.data.Dataset``` class for instantiating ```data_loader```. 
Generally, ```train_dataloader``` and ```test_dataloader``` require different Dataset classes. 
For example, in General Collaborative Filtering, we provide [PairwiseTrnData](https://github.com/HKUDS/SSLRec/blob/main/data_utils/datasets_general_cf.py) for ```train_dataloader``` to achieve negative sampling during training, 
and provide [AllRankTstData](https://github.com/HKUDS/SSLRec/blob/main/data_utils/datasets_general_cf.py) for ```test_dataloader``` to achieve All-rank evaluation.

### Model
**Model** inherits the [BasicModel](https://github.com/HKUDS/SSLRec/blob/main/models/base_model.py) class to implement various self-supervised recommendation algorithms in different scenarios.
It has four necessary functions:
+ ```__init__()```: It stores the hyper-parameter settings from user configuration as the attribute of the model, and initializes trainable parameters (e.g., user embeddings).
+ ```forward()```: It performs the model-specific forward process, such as message passing and aggregation in graph-based methods.
