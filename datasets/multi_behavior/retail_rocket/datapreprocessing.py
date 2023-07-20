import pickle
import numpy as np
from scipy.sparse import csr_matrix

predir = ''

trn_buy = pickle.load(open(predir+'trn_buy','rb'))
trn_cart = pickle.load(open(predir+'trn_cart','rb'))
trn_view = pickle.load(open(predir+'trn_view','rb'))

# #framework
# pickle.dump(trn_buy, open(predir+'train_mat_buy.pkl','wb'))
# pickle.dump(trn_cart, open(predir+'train_mat_cart.pkl','wb'))
# pickle.dump(trn_view, open(predir+'train_mat_view.pkl','wb'))
# tst_int = pickle.load(open(predir+'tst_int','rb'))
# test_x, test_y = [], []
# for index,value in enumerate(tst_int):
#     if value!=None:
#         test_x.append(index)
#         test_y.append(value)
# test_data = np.ones(len(test_x))
# # trn_buy.shape: (2174, 30113)
# test_mat = csr_matrix((test_data, (test_x, test_y)), shape=(2174, 30113))
# # pickle.dump(test_mat, open('test_mat.pkl','wb'))
# test_mat = test_mat.tocoo()
# pickle.dump(test_mat, open(predir+'test_mat.pkl','wb'))

###kmclr
# # train.txt   test.txt
# import pickle
# import pandas as pd
# import numpy as np
# trn_buy = pickle.load(open(predir+'trn_buy','rb'))
# train = 1*(trn_buy!=0)
# t_x, t_y = train[0].nonzero()
# ui_list = []
# for i in range(train.shape[0]):
#     tmp_list = [i]
#     _, tmp_item_list = train[i].nonzero()
#     tmp_list = tmp_list + tmp_item_list.tolist()
#     ui_list.append(tmp_list)

# with open(predir+'train.txt', 'w') as f1:
#     for i in range(len(ui_list)):
#         for j in range(len(ui_list[i])):
#             f1.write(str(ui_list[i][j]))
#             if(j != (len(ui_list[i])-1)):
#                 f1.write(' ')
#         f1.write('\n')
# #graph
# import pickle
# import pandas as pd
# import scipy.sparse
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
# trn_buy = pickle.load(open('trn_buy','rb'))
# graph_1 = 1*(trn_buy!=0)
# graph_2 = csr_matrix((2174, 2174))
# graph_12 = sp.hstack((graph_1, graph_2))
# graph_3 = csr_matrix((30113, 30113))
# graph_4 = graph_1.T
# graph_34 = sp.hstack((graph_3, graph_4))
# graph = sp.vstack((graph_12, graph_34))
# pickle.dump(graph, open('graph.npz','wb'))
# kg.txt
import pickle
import numpy as np
import pandas as pd
trn_view = pickle.load(open(predir+'trn_view','rb'))
trn_cart = pickle.load(open(predir+'trn_cart','rb'))
trn_buy = pickle.load(open(predir+'trn_buy','rb'))
trn_view = 1*(trn_view!=0)
trn_cart = 1*(trn_cart!=0)
trn_buy = 1*(trn_buy!=0)
ii_view = trn_view.T*trn_view
ii_cart = trn_cart.T*trn_cart
ii_buy = trn_buy.T*trn_buy
ii_view = 1*(ii_view>3)
ii_cart = 1*(ii_cart>3)
ii_buy = 1*(ii_buy>3)
view_data = np.zeros(len(ii_view.data))
cart_data = np.ones(len(ii_cart.data))
buy_data = np.full(len(ii_buy.data),2)
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





