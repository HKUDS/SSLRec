# A quick tutorial to create your own data for General Collaborative Filtering

The general collaborative filtering scenarios include three data files:

- `train_mat.pkl`: a matrix of shape `[n_user, n_item]` containing **training** interactions
- `test_mat.pkl`: a matrix of shape `[n_user, n_item]` containing **testing** interactions
- `val_mat.pkl`: a matrix of shape `[n_user, n_item]` containing **validation** interactions

All three matrices are in [coo_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) format, where each entry has a value of either 0 or 1. An entry $(i, j)=1$ indicates that there is a recorded interaction between `user_i` and `item_j`.

Suppose you have an interaction file with the following format:

```text
[1, 1, 2, 3, 4] # user_ids
[1, 2, 1, 1, 1] # item_ids
```

We can read the interaction file into two Python list objects: `user_ids` and `item_ids`. Let `n_user` and `n_item` be the total number of users and items, respectively. Then we can create the coo_matrix as follows:

```python
import numpy as np
mat = coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=[n_user, n_item])
```

We can also save the coo_matrix as a pickle file for later use. Here's an example of how to do it:

```python
import pickle
with open('mat.pkl', 'wb') as fs:
	pickle.dump(mat, fs)
```

Note that the datasets from some other scenarios in SSLRec also use a sparse matrix to store interaction data in a similar way.