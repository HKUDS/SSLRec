'''
This code is utilized to created the kg.txt file.
'''

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

predir = ''

trn_pv = pickle.load(open(predir + 'train_mat_pv.pkl', 'rb'))
trn_fav = pickle.load(open(predir + 'train_mat_fav.pkl', 'rb'))
trn_cart = pickle.load(open(predir + 'train_mat_cart.pkl', 'rb'))
trn_buy = pickle.load(open(predir + 'train_mat_buy.pkl','rb'))

# construct kg.txt
trn_pv = 1 * (trn_pv != 0)
trn_fav = 1 * (trn_fav != 0)
trn_cart = 1 * (trn_cart != 0)
trn_buy = 1 * (trn_buy != 0)

ii_pv = trn_pv.T * trn_pv
ii_fav = trn_fav.T * trn_fav
ii_cart = trn_cart.T * trn_cart
ii_buy = trn_buy.T * trn_buy

ii_pv = 1 * (ii_pv > 4)
ii_fav = 1 * (ii_fav > 3)
ii_cart = 1 * (ii_cart > 3)
ii_buy = 1 * (ii_buy > 3)

pv_data = np.zeros(len(ii_pv.data))
fav_data = np.ones(len(ii_fav.data))
cart_data = np.full(len(ii_cart.data),2)
buy_data = np.full(len(ii_buy.data), 3)

pv_x, pv_y = ii_pv.nonzero()
fav_x, fav_y = ii_fav.nonzero()
cart_x, cart_y = ii_cart.nonzero()
buy_x, buy_y = ii_buy.nonzero()


pv_kg = np.stack((pv_x, pv_data, pv_y))
fav_kg = np.stack((fav_x, fav_data, fav_y))
cart_kg = np.stack((cart_x, cart_data, cart_y))
buy_kg = np.stack((buy_x, buy_data, buy_y))

pv_kg = pv_kg.T
fav_kg = fav_kg.T
cart_kg = cart_kg.T
buy_kg = buy_kg.T

print(pv_kg.shape)
print(fav_kg.shape)
print(cart_kg.shape)
print(buy_kg.shape)

kg = np.vstack((pv_kg, fav_kg, cart_kg, buy_kg)).astype(int) 
kg_df = pd.DataFrame(kg)

kg_df.to_csv(predir + 'kg.txt', sep=' ', header=None, index=None)