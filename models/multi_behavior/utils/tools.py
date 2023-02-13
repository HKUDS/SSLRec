import random
import pandas as pd
import datetime
import numpy as np
import torch
import os
import scipy.io
import dgl
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def neg_sample(item_set,total_item,neg_sample_num):
    item_size=len(total_item)
    item=random.randint(0, item_size-1)
    item=total_item[item]
    items=[]
    # print(item_set,total_item)
    for _ in range(neg_sample_num):
        while item in item_set or item in items:
            item=random.randint(0, item_size-1)
            item=total_item[item]
        items.append(item)
    return items

def get_TIMA_size(train_dataset='dataset/TIMA/UserBehavior.csv.train.tony',test_dataset='dataset/TIMA/UserBehavior.csv.test.tony'):
    # train_data=pd.read_csv(train_dataset,names=['user_id','item_id','category_id','behavior','time'],header=None)
    # test_data=pd.read_csv(train_dataset,names=['user_id','item_id','category_id','behavior','time'],header=None)
    if train_dataset in 'dataset/TIMA/UserBehavior.csv.train':
        #987994,4162024,9439,5
        user_id=np.arange(1,987995)
        item_id=np.arange(1,4162025)
        category_id=np.arange(1,9440)    
    else:
        
        train_data=pd.read_csv(train_dataset)
        test_data=pd.read_csv(train_dataset)
    
        train_user_id=train_data['user_id'].unique()
        train_item_id=train_data['item_id'].unique()
        train_category_id=train_data['category_id'].unique()

        user_id=np.unique(np.concatenate((train_user_id,  test_data['user_id'].unique()),axis=-1))
        item_id=np.unique(np.concatenate((train_user_id,  test_data['item_id'].unique()),axis=-1))
        category_id=np.unique(np.concatenate((train_user_id,  test_data['user_id'].unique()),axis=-1))
        
    
    return user_id,item_id,category_id,5

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    AUC=0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        AUC+=(100-rank-1)/99
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
           
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list),AUC/len(pred_list)



def get_score(pred_list):
    
    pred_list=(-pred_list).argsort().argsort()[:,0]
    
    HIT_1,NDCG_1,MRR,AUC=get_metric(pred_list,20)
    HIT_5,NDCG_5,MRR,AUC=get_metric(pred_list,5)
    HIT_10,NDCG_10,MRR,AUC=get_metric(pred_list,10)
    return [HIT_1,HIT_5,HIT_10,NDCG_1,NDCG_5,NDCG_10,MRR,AUC]
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, root='',path='checkpoint', trace_func=print,saving_model=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.root=root
        self.path =self.root+path
        
        self.trace_func = trace_func
        self.saving_model=saving_model
    def __call__(self, val_loss, model,optimizer,epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer,epoch)
        
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} best score is {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
            return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer,epoch)
            self.counter = 0
            return False

    def save_checkpoint(self, val_loss, model,optimizer,epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.saving_model:
            all_states={'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'early_stop':self.counter,'loss':self.best_score}
            torch.save(all_states, self.path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.path)['net']
def get_sample_heter_graph():
    UI_PV=([1,1,1,1,2,2,2,3,3,3],[1,2,3,4,1,5,6,2,7,8])
    UI_buy=([1,2,3,1],[2,6,2,6])
    UI_cart=([1,2,3],[3,6,7])
    G = dgl.heterograph(
        {
            ('user', 'click', 'item'): UI_PV,
            ('item', 'clicked', 'user'): (UI_PV[1],UI_PV[0]),
            ('user', 'buy', 'item'): UI_buy,
            ('item', 'bought', 'user'): (UI_buy[1],UI_buy[0]),
            ('user', 'cart', 'item'): UI_cart,
            ('item', 'carted', 'user'): (UI_cart[1],UI_cart[0]),
            
        }
    )
    G_sample_edge=dgl.heterograph(
        {
            ('user', 'buy', 'item'): UI_buy,
            ('item', 'bought', 'user'): (UI_buy[1],UI_buy[0]),   
        }
    )
    G.nodes['user'].data['ID']=torch.arange(0, 4)
    G.nodes['item'].data['ID']=torch.arange(0, 9)
   
    return G


def  make_item_set(graph,vaild_set,test_set):
    item_list=[]
    user_list=[]
    for etype in ['buy','click','fav','click']:
        u,v=graph.edges(etype=etype)
        u,v=u.tolist(),v.tolist()
        item_list.extend(v)
        user_list.extend(u)
    vaild_items=[i[1] for i in vaild_set]
    test_items=[i[1] for i in test_set]
    ## 把degree 小于1的删除掉
    item_list=list(set(item_list))
    in_degrees=graph.in_degrees(torch.tensor(item_list),etype='buy')
    # in_degrees=graph.in_degrees(torch.tensor(item_list),etype='click')+ in_degrees
    item_list=torch.tensor(item_list)[in_degrees>=1]
    item_list=item_list.tolist()
    ### 加入test和vaild的item 因为他们在训练图中没有连边
    item_list.extend(vaild_items)
    item_list.extend(test_items)
    item_list=list(set(item_list))
    user_list=list(set(user_list))
    new_item_list=[]
    for item in tqdm(item_list):
        
        u,v=graph.in_edges(item,etype='click')
        # if item.item()==3534513:
        #     print(u.unique(),u.unique().size(0))
        if u.unique().size(0)!=1 and u.unique().size(0)!=0:
            new_item_list.extend(v.tolist())
    item_list=new_item_list
    item_list=list(set(item_list))
    # print(len(item_list),len(user_list))
    return item_list,user_list
def get_TIMA_Fllow_He(file='dataset/TIMA2/graph.dgl'):
    ## floow the create dataset method of He Xiangnan
    graph=dgl.load_graphs(file)
    graph=graph[0][0]
    test_set=dgl.data.utils.load_info(file+'.info')['testSet']
    for etype in ['buy','click','cart']:
        graph.nodes['item'].data[etype+'_dg']=graph.in_degrees(v='__ALL__',etype=etype)
        graph.nodes['user'].data[etype+'_dg']=graph.out_degrees(u='__ALL__',etype=etype)
    graph.nodes['item'].data['dg']=graph.in_degrees(v='__ALL__',etype='buy')+graph.in_degrees(v='__ALL__',etype='cart')+graph.in_degrees(v='__ALL__',etype='click')
    graph.nodes['user'].data['dg']=graph.out_degrees(u='__ALL__',etype='buy')+graph.out_degrees(u='__ALL__',etype='cart')+graph.out_degrees(u='__ALL__',etype='click')
   
    _,i=graph.edges(etype='buy')
    i=i.unique()
    in_dg=graph.in_degrees(i,etype='buy')
    i=i[in_dg>=1]
    item_ids=i.tolist()
    item_set=set(item_ids)
    return graph,item_ids,item_set


import _pickle as cPickle
def make_TIMA_Fllow_He(file='dataset/TIMA2/graph.dgl'):
    seq=cPickle.load(open('dataset/TIMA2/train_seq','rb'))
    

    graph=dgl.load_graphs(file)
    graph=graph[0][0]
    test_set=dgl.data.utils.load_info(file+'.info')['testSet']
    for etype in ['buy','click','cart']:
        graph.nodes['item'].data[etype+'_dg']=graph.in_degrees(v='__ALL__',etype=etype)
        graph.nodes['user'].data[etype+'_dg']=graph.out_degrees(u='__ALL__',etype=etype)
   
    _,i=graph.edges(etype='buy')
    i=i.unique()
    # print(len(test_set['U']))
    in_dg=graph.in_degrees(i,etype='buy')
    
    
    i=i[in_dg>=1]
    
    item_ids=i.tolist()
    in_dg>=1
    item_set=set(item_ids)
    test_in_dg=graph.in_degrees(test_set['I'],etype='buy')>=1
    b=[]
    for i in test_set['I']:
        if True or i in item_set:
            b.append(i)
    # print(len(set(b)))
    c=[]
    for k,v in seq.items():
        
        i=v['buy']['item_id'][-1]
        if i in item_set or True:
            c.append(i)
    # print(len(set(c)),'cc')
    test_set['U']=torch.tensor(test_set['U'])[test_in_dg].tolist()
    test_set['I']=torch.tensor(test_set['I'])[test_in_dg].tolist()
    
    vaild_graph=dgl.add_edges(graph, test_set['U'], test_set['I'],etype='buy')
    vaild_graph=dgl.add_edges(vaild_graph,  test_set['I'], test_set['U'],etype='bought')
    test_graph=dgl.add_edges(graph, test_set['U'], test_set['I'],etype='buy')
    test_graph=dgl.add_edges(test_graph,  test_set['I'], test_set['U'],etype='bought')
    return graph,vaild_graph,test_graph,item_ids


def get_TIMA_MRIG(file='dataset/TIMA2/graph4MRIG.dgl'):
    ## floow the create dataset method of He Xiangnan
    graph=dgl.load_graphs(file)
    graph=graph[0][0]
    test_set=dgl.data.utils.load_info(file+'.info')['testSet']
    a,item_ids,item_set=get_TIMA_Fllow_He()
    # print(a)
    return graph,item_ids,item_set

def get_CIKM_MRIG(file='dataset/CIKM/graph4MRIG.dgl'):
    ## floow the create dataset method of He Xiangnan
    graph=dgl.load_graphs(file)
    graph=graph[0][0]
    test_set=dgl.data.utils.load_info(file+'.info')['testSet']
    a,item_ids,item_set=get_TIMA_Fllow_He()
    # print(a)
    return graph,item_ids,item_set


def get_co_graph(file='dataset/TIMA2/cograph4MBGCN.dgl'):
    graph=dgl.load_graphs(file)
    graph=graph[0][0]
    return graph



def get_TIMA_split_traintest(file='dataset/TIMA2/graph.dgl'):
    train_file='dataset/TIMA2/traingraph.dgl'
    train_graph,item_ids,item_set=get_TIMA_Fllow_He(train_file)
    

    test_file='dataset/TIMA2/testgraph.dgl'
    test_graph,item_ids,item_set=get_TIMA_Fllow_He(test_file)
    return train_graph,test_graph,item_ids,item_set


def get_CIKM_split_traintest(file='dataset/TIMA2/graph.dgl'):
    train_file='dataset/CIKM/traingraph.dgl'
    train_graph,item_ids,item_set=get_TIMA_Fllow_He(train_file)
    

    test_file='dataset/CIKM/testgraph.dgl'
    test_graph,item_ids,item_set=get_TIMA_Fllow_He(test_file)
    return train_graph,test_graph,item_ids,item_set
   
    
