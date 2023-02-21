import pickle
import torch
import numpy as np
from scipy.sparse import csr_matrix

tima_train_x = pickle.load(open('tima_train_x','rb'))
tima_train_y = pickle.load(open('tima_train_y','rb'))
tima_test_y = pickle.load(open('tima_test_y','rb'))
tima_test_x = pickle.load(open('tima_test_x','rb'))

train_data = np.ones(tima_train_x.shape[0])
test_data = np.ones(tima_test_x.shape[0])

train_mat = csr_matrix( (train_data, (tima_train_x, tima_train_y)), shape=(22015, 27159))
test_mat = csr_matrix( (test_data, (tima_test_x, tima_test_y)), shape=(22015, 27159))
train_mat = train_mat.tocoo()
test_mat = test_mat.tocoo()
pickle.dump(train_mat, open('train_mat.pkl','wb'))
pickle.dump(test_mat, open('test_mat.pkl','wb'))
