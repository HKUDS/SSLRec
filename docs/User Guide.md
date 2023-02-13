# User Guide
The user guide contains the following content, you can quickly jump to the corresponding section

+ Architecture Design of SSLRec

## Architecture Design of SSLRec
SSLRec is a unified self-supervised recommendation algorithm framework, 
which includes the following 6 main basic classes.
### DataHandler
**DataHandler** is used to read the raw dataset, perform data preprocessing (such as converting to a sparse matrix format), and finally organize the data into a DataLoader for training and testing.
In our design, it contains two important functions:
+ ```__init__()```: It stores the original path of the corresponding dataset according to the configuration provided by the user.
+ ```load_data()```: It read the raw dataset data, perform necessary data preprocessing and finally instantiate ```train_dataloader``` and ```test_dataloader```

We designed the respective DataHandler for the four categories (i.e., General Collaborative Filtering, Sequential Recommendation, Multi-behavior Recommendation, 
Social Recommendation). You can get a more detailed understanding by reading the source code of [DataHandlerGeneralCF](https://github.com/HKUDS/SSLRec/blob/main/data_utils/data_handler_general_cf.py).