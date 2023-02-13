# User Guide
The user guide contains the following content, you can quickly jump to the corresponding section.

+ Architecture Design of SSLRec

## Architecture Design of SSLRec
SSLRec is a unified self-supervised recommendation algorithm framework, 
which includes the following 5 main parts.
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
and provide [AllRankTstData](https://github.com/HKUDS/SSLRec/blob/main/data_utils/datasets_general_cf.py) for ```test_dataloader``` to achieve all-rank evaluation.

### Model
**Model** inherits the [BasicModel](https://github.com/HKUDS/SSLRec/blob/main/models/base_model.py) class to implement various self-supervised recommendation algorithms in different scenarios.
It has four necessary functions:
+ ```__init__()```: It stores the hyper-parameter settings from user configuration as the attribute of the model, and initializes trainable parameters (e.g., user embeddings).
+ ```forward()```: It performs the model-specific forward process, such as message passing and aggregation in graph-based methods.
+ ```cal_loss(batch_data)```: The input ```batch_data (tuple)``` is a batch of training samples provided by ```train_loader```. 
  This function calculates the loss function defined by the model and has two return values: (1) ```loss (0-d torch.Tensor)``` : the overall weighted loss, (2) ```losses (dict)``` dict for specific terms of losses for printing.
+ ```full_predict(batch_data)```: The input ```batch_data (tuple)``` is the data in a test batch (e.g., ```batch_users``` (the tested users in this batch) and ```train_mask``` (training items of those users)). 
  This function return a prediction tensor ```full_pred (torch.Tensor)``` for all-rank evaluation.

You can get a more detailed understanding by reading the source code of [LightGCN](https://github.com/HKUDS/SSLRec/blob/main/models/general_cf/lightgcn.py).

### Trainer
**Trainer** provides a unified process of training, testing and storing model parameters. 
Using a unified trainer for different models can ensure the fairness of comparison. Our trainer including the following six functions:
+ ```create_optimizer(model)```: It creates the optimizer (e.g., ```torch.optim.Adam```) according to the configuration.
+ ```train_epoch(model, epoch_idx)```: It performs one epoch training, including calculating loss, optimizing parameters and printing the losses.
+ ```save_model(model)```: It saves the model parameters as a ```pth``` file.
+ ```load_model(model)```: It loads the model parameters from a ```pth``` file.
+ ```evaluate(model)```: It evaluates the model on test/validation set and return the results of selected metrics according to the configuration.
+ ```train(model)```: It conducts the whole training, testing and saving process.

Sometimes, some models may use different training process during one epoch. 
We recommend only overwriting the ```train_epoch(model, epoch_idx)``` to ensure a fair comparison.
You can read [Create My Own Trainer]() for more details.

### Configuration
Each model has its own different configuration, we write it in a ```yml``` file (e.g., [lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml))
In a ```yml``` file, the following keys are required:
+ ```optimizer```: It contains necessary information to create an optimizer, such as the name of that optimizer and learing rate.
+ ```train```: It contains the setting of training process, such as the number of epochs, the size of each batch and so on.
+ ```test```: It sets the necessary configuration for testing, such as metrics, etc.
+ ```data```: It determines which dataset to use.
+ ```model```: It determines which model to create and the hyper-parameters of that model.

If you create your own model, then you have to create a configuration file for it. We recommend you to read 
[lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml) to get a basic impression of how to write configuration files, 
then jump to [Create My Own Configuration](), in which we provided a more detailed description.