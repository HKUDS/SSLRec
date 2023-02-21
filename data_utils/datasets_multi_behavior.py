import numpy as np
import collections
from scipy.sparse import coo_matrix, dok_matrix, csr_matrix
import pandas as pd
from tqdm import tqdm
import datetime
import random
import json
import dgl
# from dgl.data import DGLDataset
import pickle
from time import time
import scipy.sparse as sp


import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from config.configurator import configs
from models.multi_behavior.utils import tools



class AllRankTestData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        user_pos_lists = [list() for i in range(coomat.shape[0])]
        # user_pos_lists = set()
        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class PairwiseTrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
	
	def sample_negs(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(configs['data']['item_num'])
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
	
	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]



class CMLData(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        for i in range(self.length):
            self.neg_data[i] = [None] * len(self.beh)
            self.pos_data[i] = [None] * len(self.beh)

        for index in range(len(self.beh)):

            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index].tocsr()[uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index].tocsr()[uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i



class MMCLRData(Dataset):
    def __init__(self, neg_sample_num=1,root_dir='MMCLR/dataset/TIMA/UserBehavior.10%.seq.splited.pickle',eval=None):
        
        super(MMCLRData, self).__init__()
        self.root_dir=root_dir
        self.eval=eval
        self.item_set=set(configs['model']['item_ids'])
        self.count=0
        # print(root_dir)
        self.eavl=eval
        if eval is None :
            self.data=self.read_data(self.root_dir)
        else:
            self.data=self.read_data_eval(self.root_dir)
        
        self.rng=random.Random(configs['train']['random_seed'])
        self.neg_sample_num=neg_sample_num
        self.raw_data=self.make_raw_data(root_dir)
        self.hardSet=self.make_hard_sample_item_set(root_dir)


    def make_hard_sample_item_set(self,file):
        f=open(file,'rb')
        all_seq={}
        b_id_set=[]
        all_info=pickle.load(f)
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            bs=['pv','cart']
            
            time=buy_times[-1]
            for b in bs:
                if len(user_info[b]['item_id'])==0:
                    continue
                b_ids=np.array(user_info[b]['item_id'])
                b_times=np.array(user_info[b]['times'])
                index=b_times>time
                pos_b_ids=b_ids[index]
                b_ids=b_ids[~index]
                pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]
                b_id_set.extend(pos_b_ids)
                
        # print(len(set(b_id_set)))
        return set(b_id_set)
    def make_raw_data(self,file): 
        f=open(file,'rb')
        all_seq={}
        all_info=pickle.load(f)
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            bs=['fav','pv','cart']
            
            if len(buy_ids)==1 and self.eavl is None:## 如果少于2（不在训练集中）
                # one_seq['fav']=[]
                # one_seq['pv']=[]
                # one_seq['cart']=[]
                # one_seq['buy']=buy_ids
                # all_seq[user]=one_seq
                continue
           
            if self.eavl is None:
                time=buy_times[-2]## 训练集中最后购买的时间戳
                buy_sub_item_ids=buy_ids[:-1]
            else:
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids  
            
            for b in bs:
                if b=='buy':
                    continue
                if b not in user_info or len(user_info[b]['item_id'])==0:
                    b_ids=[]
                    pos_b_ids=[]
                else:
                    b_ids=np.array(user_info[b]['item_id'])
                    b_times=np.array(user_info[b]['times'])
                    index=b_times>time
                    pos_b_ids=b_ids[index].tolist() ## 后面的行为有多少
                    index=b_times<=time
                    b_ids=b_ids[index].tolist() #前面的行为
                    
                one_seq[b]=b_ids
                one_seq['pos'+b]=pos_b_ids
                one_seq['buy']=buy_sub_item_ids
            
            all_seq[user]=one_seq
        return all_seq
            
    def read_data_eval(self,file):
        f=open(file,'rb')
        
        all_info=pickle.load(f)
        
        f.close()
        all_seq=[]
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            one_seq['posbuy']=[buy_ids[-1]]
            # if len(buy_ids)<4:
            #     continue
            
            if self.eval=='vaild':
                if buy_ids[-2] not in self.item_set:
                    
                    continue
                if buy_ids[-2] in buy_ids[:-2]:
                    
                    continue
                time=buy_times[-2]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='test':
                
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='cold_start':
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                if len(buy_ids)>3:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='uncold_start':
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                if len(buy_ids)<=3:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times

            else:
                time=buy_times[-2]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            
            bs=['fav','pv','cart']
            for b in bs:
                if b=='buy':
                    continue
                if b not in user_info or len(user_info[b]['item_id'])==0:
                    b_ids=[]
                    pos_b_ids=[]
                else:
                    b_ids=np.array(user_info[b]['item_id'])
                    b_times=np.array(user_info[b]['times'])
                    index=b_times>time
                    pos_b_ids=b_ids[index].tolist()
                    
                    index=b_times<=time
                    b_ids=b_ids[index].tolist()
                    pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]

                one_seq[b]=b_ids
                one_seq['pos'+b]=pos_b_ids
            one_seq['pospv']=[later_item for later_item in one_seq['pospv'] if later_item not in one_seq['cart']]
            one_seq['poscart']=[later_item for later_item in one_seq['poscart'] if later_item not in one_seq['pv']]
            one_seq['buy']=buy_sub_item_ids
            
            if len(buy_sub_item_ids)==1 and len(one_seq['cart'])==0 and len(one_seq['pv'])==0: ## if only have one buy behavior
                    continue
            all_seq.append(one_seq)
        return all_seq
                

    def read_data(self,file):
        f=open(file,'rb')
        
        all_info=pickle.load(f)
        f.close()
        all_seq=[]
        
        for user,user_info in tqdm(all_info.items()):
           
            buy_ids=user_info['buy']['item_id']
            
            buy_times=user_info['buy']['times']
            # if len(buy_ids)<4:
            #     continue
            for i,item_id in enumerate(buy_ids[:-1]):
                
                if item_id not in self.item_set:
                    # print(user,item_id)
                    continue
                
                one_seq={'user_id':user}
                time=buy_times[i]
                next_time=buy_times[i+1]
                buy_sub_item_ids=buy_ids[:i+1]
                one_seq['posbuy']=buy_ids[i+1:-1]
                buy_sub_times=buy_times[:i+1]
                bs=['fav','pv','cart']
                for b in bs:
                    if b=='buy':
                        continue
                    if b not in user_info:
                        b_ids=[]
                        pos_b_ids=[]
                    else:

                        b_ids=np.array(user_info[b]['item_id'])
                        pos_b_ids=np.array([])
                        if len(b_ids)!=0: ##当有该行为的时候
                            
                            b_times=np.array(user_info[b]['times'])
                            index=(b_times>time) & (b_times<=next_time)
                            

                            pos_b_ids=b_ids[index]
                            
                            index=b_times<=time
                            b_ids=b_ids[index]
                            
                        
                        b_ids=b_ids.tolist()
                        
                       
                        pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]

                    one_seq[b]=b_ids
                    one_seq['pos'+b]=pos_b_ids
                one_seq['buy']=buy_sub_item_ids
                if len(buy_sub_item_ids)==1 and len(one_seq['cart'])==0 and len(one_seq['pv'])==0: ## if only have one buy behavior
                    continue
                all_seq.append(one_seq)
        return all_seq
                
    def __len__(self):
        return len(self.data)

    def encode_behavior(self,behvaior):
        be2code={'pv':1,'cart':2,'fav':3,'buy':4}
        return be2code[behvaior]
    
    def mask_seq(self,mask_item):
        masked_item_seq=[]
        negtive_seq=[]
        mask_num=1
        for i in mask_item[:-1]:
                prob=self.rng.random()
                if prob<configs['train']['mask_prob']:
                    prob=prob/configs['train']['mask_prob']
                    
                    if prob < 0.8:
                        mask_num+=1
                        masked_item_seq.append(configs['model']['mask_id'])
                        neg=tools.neg_sample(set(mask_item), configs['model']['item_ids'], self.neg_sample_num)
                        negtive_seq.append(neg[0]) 
                    elif prob < 0.9:
                        mask_num+=1
                        masked_item_seq.append(self.rng.randint(1,configs['model']['item_size']-4))
                        # negtive_seq.append(tools.neg_sample(set(buy_item_seq), configs['model']['item_ids'], self.neg_sample_num))
                        neg=tools.neg_sample(set(mask_item), configs['model']['item_ids'], self.neg_sample_num)
                        negtive_seq.append(neg[0])
                    else:
                        masked_item_seq.append(i)
                        # masked_cate_seq.append(c)
                        negtive_seq.append(i)
                else:
                        masked_item_seq.append(i)
                        # masked_cate_seq.append(c)
                        negtive_seq.append(i)

       
        pos_seq=mask_item
        negtive_seq.append(tools.neg_sample(set(mask_item), configs['model']['item_ids'], self.neg_sample_num)[0])
        masked_item_seq.append(configs['model']['mask_id'])
        
        return masked_item_seq,pos_seq,negtive_seq,mask_num
    def __getitem__(self,index):
        user_id=self.data[index]['user_id']
    
        pv_item_seq=[configs['model']['start_id']]+self.data[index]['pv']+[configs['model']['end_id']]
      
        buy_item_seq=self.data[index]['buy']
        fav_item_seq=[configs['model']['start_id']]+self.data[index]['buy']+[configs['model']['end_id']]
        cart_item_seq=[configs['model']['start_id']]+self.data[index]['cart']+[configs['model']['end_id']]
        pos_buy_item_seq=[configs['model']['start_id']]+self.data[index]['posbuy']+[configs['model']['end_id']]
        ### get constractive sample
        multi_items=[self.data[index]['buy'],self.data[index]['pv'],self.data[index]['cart']]
        have_constra=1
        have_click=1 ## have cart for inner cons
        if len(multi_items[0])==0 or len(multi_items[1])==0 or len(multi_items[2])==0:
                      
            b=[ i for i,j in enumerate(multi_items) if  len(j)>0]
            if len(b)>1:
                
                b3=b
                c=self.rng.randint(0, 1)
                b1=b3[c]
                b2=b3[1-c]
            else:
                have_constra=0
                have_click=0
                b3=[0,1]
                b1,b2=0,0
            if 1 not in b:
                have_click=0
        else:
            have_cart=1         
            b1,b2,b3=0,1,[0]
            b1=self.rng.randint(0, 2)
            b2=b1
            while b2==b1:
                b2=self.rng.randint(0, 2)
            b3=[b1,b2]
        b1,b2=multi_items[b1][-configs['model']['max_seq_len']:],multi_items[b2][-configs['model']['max_seq_len']:]
        if b1==0:
            b1=([configs['model']['start_id']]+b1[-configs['model']['max_seq_len']+2:]+[configs['model']['end_id']])
        if b2==0:
            b2=([configs['model']['start_id']]+b2[-configs['model']['max_seq_len']+2:]+[configs['model']['end_id']])
        con_len=[len(b1),len(b2),len(b3)]
        b1=[0]*(configs['model']['max_seq_len']-len(b1))+b1
        b2=[0]*(configs['model']['max_seq_len']-len(b2))+b2
        # if len(b1)!=len(b2):
        #     print(len(b1),len(b2),b3)
        behavior_ctra_sample=(b1,b2)
        if self.eval is None:
            ## here we only mask last item 
            masked_item_seq,pos_seq,negtive_seq,mask_num=self.mask_seq(buy_item_seq)
        else:
            mask_num=1
            if self.eval =='test' or self.eval=='cold_start' or self.eval=='uncold_start':
                pos_seq=[buy_item_seq[-1]]
                masked_item_seq=buy_item_seq[:-1]
            elif self.eval=='vaild':
                pos_seq=[buy_item_seq[-2]]
                masked_item_seq=buy_item_seq[:-2]
            
            negtive_seq=tools.neg_sample(set(buy_item_seq), configs['model']['item_ids'], self.neg_sample_num)
            masked_item_seq.append(configs['model']['mask_id'])
        ## here we smaple the click for innerCL
        sampled_clicks=[-1]*50
        sample_item=self.data[index]['pospv']
        aragen=len(sample_item) # pospv是当前预测的 buy item time 之后到下一个 buy item之前的所有click item
        for i in range(mask_num):
            if aragen==0 :
                sampled_clicks[i]=0 #当sampleclick 没有时候
                continue
            aragen=min(len(sample_item),10) ##如果太多只取10个
            sampled_click=self.rng.randint(0, aragen-1)
            sampled_click=sample_item[sampled_click]
            sampled_clicks[i]=sampled_click
        sample_items=self.data[index]['poscart']+self.data[index]['pospv']
        if self.eavl:
            if aragen!=0:
                aragen=min(len(sample_items),10)
                for i in range(aragen):
                    sampled_clicks[i]=sample_items[i]
        pad_len=configs['model']['max_seq_len']-len(masked_item_seq)
        masked_item_seq=masked_item_seq[-configs['model']['max_seq_len']:]
        masked_item_seq=[0]*pad_len+masked_item_seq
        pad_len=configs['model']['max_seq_len']-len(pv_item_seq)
        pv_item_seq=[0]*pad_len+pv_item_seq[-configs['model']['max_seq_len']:]
        pad_len=configs['model']['max_seq_len']-len(cart_item_seq)
       
        cart_item_seq=[0]*pad_len+cart_item_seq[-configs['model']['max_seq_len']:]
        pad_len=configs['model']['max_seq_len']-len(fav_item_seq)
        fav_item_seq=[0]*pad_len+fav_item_seq[-configs['model']['max_seq_len']:]
        pad_len=configs['model']['max_seq_len']-len(pos_buy_item_seq)
        pos_buy_item_seq=[0]*pad_len+pos_buy_item_seq[-configs['model']['max_seq_len']:]               
        if self.eval is None:
            pad_len=configs['model']['max_seq_len']-len(pos_seq)
            pos_seq=[0]*pad_len+pos_seq[-configs['model']['max_seq_len']:]        
            negtive_seq=[0]*pad_len+negtive_seq[-configs['model']['max_seq_len']:]
        cur_tensor=(
            torch.LongTensor([user_id]),
            torch.LongTensor(masked_item_seq),
            torch.LongTensor(pv_item_seq),
            torch.LongTensor(cart_item_seq),
            torch.LongTensor(fav_item_seq),
            torch.tensor(pos_seq,dtype=torch.long),
            torch.tensor(negtive_seq,dtype=torch.long),
            torch.tensor(b1,dtype=torch.long),
            torch.tensor(b2,dtype=torch.long),
            torch.tensor(b3,dtype=torch.long),
            torch.tensor([have_click],dtype=torch.long),
            torch.tensor(sampled_clicks,dtype=torch.long),
            torch.tensor([have_constra],dtype=torch.long)
        )
        return cur_tensor


class HMGCRData(data.Dataset):
	def __init__(self, data, num_item, train_mat=None, num_ng=0, is_training=None):
		super(HMGCRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.data = np.array(data)
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def sample_negs(self):
		# assert self.is_training, 'no need to sampling when testing'
		tmp_trainMat = self.train_mat.todok()
		length = self.data.shape[0]
		self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

		for i in range(length):
			uid = self.data[i][0]
			iid = self.neg_data[i]
			if (uid, iid) in tmp_trainMat:
				while (uid, iid) in tmp_trainMat:
					iid = np.random.randint(low=0, high=self.num_item)
				self.neg_data[i] = iid

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		user = self.data[idx][0]
		item_i = self.data[idx][1]
		if self.is_training:
			neg_data = self.neg_data
			item_j = neg_data[idx]
			return user, item_i, item_j 
		else:
			return user, item_i




class KMCLRData(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):  
        super(KMCLRData, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0] 
        self.neg_data = [None]*self.length  
        self.pos_data = [None]*self.length  

    def ng_sample(self):
        for i in range(self.length):
            self.neg_data[i] = [None]*len(self.beh)
            self.pos_data[i] = [None]*len(self.beh)

        for index in range(len(self.beh)):
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh)-1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index].tocsr()[uid].data)==0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index].tocsr()[uid].toarray()
                        pos_index = np.where(t_array!=0)[1]
                        iid_pos = np.random.choice(pos_index, size = 1, replace=True, p=None)[0]   
                        self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:  
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:  
            return user, item_i



class KGDataset(Dataset):
    def __init__(self, m_item, kg_path = './datasets/' + configs['data']['name'] + '/kg.txt'):
        self.m_item = m_item
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)

    @property
    def entity_count(self):
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        return self.kg_data['r'].max() + 2

    def get_kg_dict(self, item_num):
        entity_num = configs['model']['entity_num_per_item'] 
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))
                if (len(tails) > entity_num):
                    i2es[item] = torch.IntTensor(tails).to(configs['model']['device'])[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(configs['model']['device'])[:entity_num]
                else:
                    tails.extend([self.entity_count] * (entity_num - len(tails)))
                    relations.extend([self.relation_count] * (entity_num - len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(configs['model']['device'])
                    i2rs[item] = torch.IntTensor(relations).to(configs['model']['device'])
            else:
                i2es[item] = torch.IntTensor([self.entity_count] * entity_num).to(configs['model']['device'])
                i2rs[item] = torch.IntTensor([self.relation_count] * entity_num).to(configs['model']['device'])
        return i2es, i2rs


    def generate_kg_data(self, kg_data):
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            if h >= self.m_item:
                continue
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads


    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError



class UIDataset(BasicDataset):
    def __init__(self, path):
        self.split = configs['model']['A_split']
        self.folds = configs['model']['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + 'train.txt'
        test_file = path + 'test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.m_item += 1
        self.n_user += 1
        self.Graph = None

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),dtype=np.float32)

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

    @property
    def item_groups(self):
        with open(self.path + "/item_groups.pkl", 'rb') as f:
            g = pickle.load(f)
        return g

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(
                    configs['model']['device']))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/graph.npz')
                norm_adj = pre_adj_mat
            except:
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items),
                    dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(self.path + '/graph.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(configs['model']['device'])
        return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users,
                                         items]).astype('uint8').reshape(
            (-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg




#----Sampler------------------------------------------------------------------------------------------------------------------------


class MMCLRNeighborSampler(object):
    def __init__(self, g, num_layers,neg_sample_num=1,is_eval=False):
        self.g = g
        self.num_layers=num_layers
        self.is_eval=is_eval
        self.neg_sample_num=neg_sample_num
        self.rng=random.Random(configs['train']['random_seed'])
        self.error_count=0
        self.total=0

    

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        block_src_nodes=[]
        for layer in range(self.num_layers):
            frontier = dgl.in_subgraph(self.g, seeds)
            frontier=dgl.compact_graphs(frontier,always_preserve=seeds)
            seeds = frontier.srcdata[dgl.NID]
            blocks.insert(0, frontier)
            src_nodes={}
            for ntype in frontier.ntypes:
                src_nodes[ntype]=frontier.nodes(ntype=ntype)
            block_src_nodes.insert(0,src_nodes)
        input_nodes=seeds
       

        return input_nodes,blocks,block_src_nodes

    def sample_neg_user(self,batch_users):
        batch_users=batch_users.tolist()
        neg_batch_users=[]
        user_set=list(set(batch_users))
        
        for user in batch_users:
            neg_user=user
            while neg_user==user:
                neg_user_idx=self.rng.randint(0,len(user_set)-1)
                neg_user=user_set[neg_user_idx]
                
            neg_batch_users.append(neg_user_idx)
        return torch.tensor(neg_batch_users)

    def sample_from_item_pairs(self, seq_tensors):
        neg_src=[]
        pos_src=[]
        pos_dst=[]
        neg_dst=[]
        batch_tensors=[[] for _ in range(len(seq_tensors[0]))]
        for seq in seq_tensors:
            user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,b3,con_len,sampled_click,pos_buy_item_seq=seq
            
            for i,data in enumerate(seq):
                batch_tensors[i].append(data)
            if self.is_eval:

                neg_dst.append(neg)
                pos_dst.append(pos)
            else:
                masked=pos-neg
                pos_dst.append(pos[masked!=0])
                neg_dst.append(neg[masked!=0])
            pos_src.append(user_id.repeat(pos_dst[-1].shape[0]))
            neg_src.append(user_id.repeat(neg_dst[-1].shape[0]))
       
        batch_tensors=[ torch.stack(tensors,dim=0) for tensors in batch_tensors]
        batch_tensors[0]=batch_tensors[0].reshape(-1)
        neg_user_ids=torch.tensor([0])
        pos_dst=torch.cat(pos_dst,axis=0)
        neg_dst=torch.cat(neg_dst,axis=0)
        neg_src=torch.cat(neg_src,axis=0)
        pos_src=torch.cat(pos_src,axis=0)
        
        pos_graph = dgl.heterograph({('user','buy','item'):
            (pos_src, pos_dst),
           
            
            }
            )
        neg_graph = dgl.heterograph(
            {('user','buy','item'):
            (neg_src, neg_dst),
            }
            )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        buy_items=torch.cat((pos_dst,neg_dst),dim=0)
        seeds = {'user':batch_tensors[0],'item':torch.cat((pos_dst,neg_dst),dim=0)}
        input_nodes,blocks,block_src_nodes = None,None,None
        return input_nodes,pos_graph, neg_graph, blocks,block_src_nodes,batch_tensors,neg_user_ids



