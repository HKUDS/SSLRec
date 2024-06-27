# Datasets
In this document, we provide detailed descriptions of the data files for each scenario.

## General Collaborative Filtering
- `train_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]`. It represents the training set interactions, where each entry with a value greater than 1 indicates an interaction.
- `valid_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]`. It represents the validation set interactions, where each entry with a value greater than 1 indicates an interaction.
- `test_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]`. It represents the test set interactions, where each entry with a value greater than 1 indicates an interaction.

## Sequential Recommendation
- `train.tsv`: This file contains the training sessions. Each line of record follows the format `[session_id, item_id_seq, item_id]`. The `item_id_seq` is the observed sequence and the `item_id` is the next item for supervision.
- `test.tsv` This file contains the testing sessions. Each line of record follows the format `[session_id, item_id_seq, item_id]`. The `item_id_seq` is the observed sequence and the `item_id` is the ground truth next item for testing.

## Social Recommendation
- `trn_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]`. It represents the training set interactions, where each non-zero entry indicates an interaction.
- `trn_time.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]` contains timestamps (matrix entry values) of interactions in the training set.
- `trust_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_user]` represents social relationships, where non-zero entries indicate that two users have a social relationship.
- `test_mat.pkl` This file contains a SciPy sparse matrix of shape `[n_user, n_item]`. It represents the test set interactions, where each entry with a value greater than 1 indicates an interaction.
- `category.pkl` This file contains a SciPy sparse matrix of shape `[n_item, n_category]` represents item categories, where entries mark the categories that an item belongs to (possibly multiple).

## Knowledge Graph-enhanced Recommendation
- `train.txt` This file contains the observed user-item interactions for training. Each line starts with a User ID, followed by the IDs of items that the user has interacted with.
- `test.txt` This file contains the user-item interactions for testing. Each line starts with a User ID, followed by the IDs of items that the user has interacted with.
- `kg_final.txt` This file contains the KG tripltes. Each line of the record follows the format `[item_id, relation_id, entry_id]`.
- `item_list.txt` This file contains the item ID information, which includes `org_id`, `remap_id` and `freebase_id`.
- `entity_list.txt` This file contains the entity ID information, which includes `freebase_id` and `remap_id`.
- `user_list.txt` This file contains the user ID information, which includes `org_id` and `remap_id`
- `relation_list.txt` This file contains the relation ID information, which includes `org_id` and `remap_id`

## Multi-behavior Recommendation
- `train_mat_{behavoir}.pkl (e.g., train_mat_buy.pkl)` This file contains a SciPy sparse matrix with a shape of `[n_user, n_item]`, where each non-zero entry represents an interaction under a specific behavior.
- `train_mat_{meta_path}.pkl (e.g.,train_mat_pv_buy.pkl)` This file contains a SciPy sparse matrix with a shape of `[n_user, n_item]`. Each non-zero entry in the matrix indicates an interaction between a user and an item. The interactions recorded in this matrix are the intersection of multiple behaviors.
- `test_mat.pkl` This file contains a SciPy sparse matrix with a shape of `[n_user, n_item]`. It represents the interactions for evaluation in the test set, where each non-zero entry represents an interaction.
- `meta_multi_single_beh_user_index_shuffle` This file is a list that contains the IDs of users who exhibit specific behaviors used for the meta-training of the `CML` model (e.g., active users, to avoid model overfitting on noise data).
- `kg.txt (from datapreprocessing.py)` This file contains records of item-item relationships for the `KMCLR` model, where each interaction is associated with a specific behavior. Each line of the record follows the format `[item_id, behavior_id, item_id]`, indicating the interaction between two items with the corresponding behavior.