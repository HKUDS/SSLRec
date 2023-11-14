'''
This code is utilized to created the kg.txt file.
'''

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

predir = ''

trn_buy = pickle.load(open(predir+'train_mat_buy.pkl', 'rb'))
trn_cart = pickle.load(open(predir+'train_mat_cart.pkl', 'rb'))
trn_view = pickle.load(open(predir+'train_mat_view.pkl', 'rb'))

# construct kg.txt
trn_view = 1 * (trn_view != 0)
trn_cart = 1 * (trn_cart != 0)
trn_buy = 1 * (trn_buy != 0)

ii_view = trn_view.T * trn_view
ii_cart = trn_cart.T * trn_cart
ii_buy = trn_buy.T * trn_buy

ii_view = 1 * (ii_view > 3)
ii_cart = 1 * (ii_cart > 3)
ii_buy = 1 * (ii_buy > 3)

view_data = np.zeros(len(ii_view.data))
cart_data = np.ones(len(ii_cart.data))
buy_data = np.full(len(ii_buy.data), 2)

view_x, view_y = ii_view.nonzero()
cart_x, cart_y = ii_cart.nonzero()
buy_x, buy_y = ii_buy.nonzero()

view_kg = np.stack((view_x, view_data, view_y))
cart_kg = np.stack((cart_x, cart_data, cart_y))
buy_kg = np.stack((buy_x, buy_data, buy_y))

view_kg = view_kg.T
cart_kg = cart_kg.T
buy_kg = buy_kg.T

print(view_kg.shape)
print(cart_kg.shape)
print(buy_kg.shape)

kg = np.vstack((view_kg, cart_kg, buy_kg)).astype(int) 
kg_df = pd.DataFrame(kg)
# kg_df.to_csv(predir+'kg.txt', sep=' ', header=None, index=None)





