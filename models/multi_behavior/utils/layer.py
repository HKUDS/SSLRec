import math
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl

from models.multi_behavior.bert_modules.bert import BERT
from models.multi_behavior.bert_modules.utils import LayerNorm
from models.multi_behavior.utils import tools
from models.multi_behavior.bert_modules.embedding.token import TokenEmbedding

from config.configurator import configs


class SequenceLayer(torch.nn.Module):
    def __init__(self, item_embedding_layer):
        super(SequenceLayer, self).__init__()
        self.token = item_embedding_layer
        self.buy_bert = BERT()
        self.cart_bert=BERT()
        self.fav_bert=BERT()
        self.pv_bert=BERT()
        self.fc_user=nn.Sequential(nn.Linear(configs['model']['embedding_size']*3, configs['model']['embedding_size']),
                                    nn.ReLU(True),
                                    nn.Linear(configs['model']['embedding_size'],configs['model']['embedding_size'])
                                )
        self.fc_item=nn.Sequential(nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']*2),
                                    nn.ReLU(True),
                                    nn.Linear(configs['model']['embedding_size']*2,configs['model']['embedding_size'])
                                )
        self.PRLoss=torch.nn.BCELoss()
        self.RNN=torch.nn.GRU(input_size=configs['model']['embedding_size'],hidden_size=configs['model']['embedding_size'],batch_first=True)
        self.weight=nn.Parameter(torch.FloatTensor([0.1,0.9]))
        nn.init.uniform_(self.weight)
        self.fc2=nn.Sequential(
            nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']),
            #  nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']),
            # nn.LayerNorm(configs['model']['embedding_size'])
            
            

        )
        self.fc3=nn.Sequential(
           nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']),

        )
        
        self.rng=random.Random(configs['train']['random_seed'])
    def neg_sample_user_batch(self,user_ids):
        user_ids=user_ids.tolist()
        neg_b=[]
        for user_id in user_ids:
            index=self.rng.randint(0,len(user_ids)-1)
            while user_ids[index]==user_id:
                index=self.rng.randint(0,len(user_ids)-1)
            neg_b.append(index)
        return neg_b

    def inner_triple(self,buy_seq,click_emb,cart_emb,have_cart,is_eavl=False):
      
        buy_emb=self.token(buy_seq)
        buy_emb=self.buy_bert(buy_seq,buy_emb)[:,-1,:]
        mask=have_cart==1
        click_emb=click_emb[:,-1,:]
        cart_emb=cart_emb[:,-1,:]
        click_emb_masked=click_emb[mask]
        cart_emb_masked=cart_emb[mask]
        buy_emb_masked=buy_emb[mask]
        buy_emb_transed=self.fc2(buy_emb_masked)
        click_emb_transed=self.fc2(click_emb_masked)
        cart_emb_transed=self.fc2(cart_emb_masked)
        a=(buy_emb_transed*cart_emb_transed).sum(dim=1)
        b=(buy_emb_transed*click_emb_transed).sum(dim=1)
        inner_loss=(b-a).sigmoid()
        inner_loss=inner_loss.clamp(max=0.7)
        inner_loss=self.PRLoss(inner_loss,torch.ones_like(inner_loss))        
        return inner_loss,(buy_emb,click_emb)

    def constras_behavior_seq(self,b1,b2,b3,user_ids,have_constra):
        b1_emb_seq,b2_emb_seq=self.token(b1),self.token(b2)
        b2_emb=self.buy_bert(b2,b2_emb_seq)[:,-1,:]
        b1_emb=self.buy_bert(b1,b1_emb_seq)[:,-1,:]
        b2_emb_transed=self.fc2(b2_emb)
        b1_emb_transed=self.fc2(b1_emb)
        noconstra_mask=have_constra!=0
        mask=(user_ids.unsqueeze(0)==user_ids.unsqueeze(1))
        mask=~mask
        mask=mask[noconstra_mask] ##仅仅取有con行为的mask mask shap是 have_constr_num *Batch_size
        scores=torch.matmul(b1_emb_transed,b2_emb_transed.T)
        
        self_scores=torch.diag(scores).reshape(-1,1)
        scores=scores[noconstra_mask]
        self_scores=self_scores[noconstra_mask]
        scores=(self_scores-scores).sigmoid()
        
        scores=scores[mask]
     
        loss=self.PRLoss(scores,torch.ones_like(scores,dtype=torch.float32))
        return loss,(b1_emb,b2_emb),scores,0
      

    def forward(self, batch,is_eval):
        user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,b3,have_cart,sampled_click,have_constra=batch
        sampled_click=sampled_click[sampled_click!=-1]
        masked_item_embedding=self.token(masked_item_seq)
        seq_len=masked_item_seq.size(1)
        pv_seq_embedding=self.token(pv_item_seq)
        cart_seq_embedding=self.token(cart_item_seq)
        masked_item_embedding = self.buy_bert(masked_item_seq,masked_item_embedding)#B*L*H
        pv_embedding= self.buy_bert(pv_item_seq,pv_seq_embedding)[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)#B*H
        cart_embedding= self.buy_bert(cart_item_seq,cart_seq_embedding)[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)#B*H
        x=torch.cat([masked_item_embedding,pv_embedding,cart_embedding],axis=2)
        sampled_click_emb=self.token(sampled_click).reshape(-1,configs['model']['embedding_size'])
        if not is_eval:
            x=x.view(-1,x.shape[-1]) #B*LxH
            x=self.fc_user(x)
            pos=self.token(pos).view(-1,configs['model']['embedding_size']) #B*LxH
            neg=self.token(neg).view(-1,configs['model']['embedding_size']) #B*LxH
        if is_eval:
            x=x[:,-1,:] #B*H
            x=self.fc_user(x)
            pos=self.token(pos).reshape(-1,configs['model']['embedding_size']) #B x H
            neg=self.token(neg).reshape(-1,configs['model']['embedding_size'])#B*negNum x H
        ce_loss,click_seq_emb,scores,label=self.constras_behavior_seq(b1,b2,b3,user_id,have_constra=have_constra.squeeze(-1))
        inner_loss,buy_click_emb=self.inner_triple(fav_item_seq,pv_embedding, cart_embedding,have_cart.squeeze(-1),is_eavl=is_eval)
        return x,pos,neg,inner_loss,ce_loss,buy_click_emb,sampled_click_emb
    

class HeterLightGCNLayer(torch.nn.Module):
    def __init__(self,op_edges,res=True):
        super(HeterLightGCNLayer, self).__init__()
        self.op_edges=op_edges
        self.dropout=torch.nn.Dropout()
    def message_func(self,edges):

        return {'m':edges.src['normed']}
    def reduce_func(self,nodes):
        # print( torch.sum(nodes.mailbox['m'],dim=1).shape)
        return {'n_f': torch.sum(nodes.mailbox['m'], dim=1)}
    def forward(self,G,h):
        rst_dic={}
        funcs={}
        
        with G.local_scope():
            for src_type,etype,dst_type in G.canonical_etypes:
                
                # g_degree=G.out_degrees(u='__ALL__',etype=etype)
                g_degree=G.nodes[src_type].data[self.op_edges[etype]+'_dg']
                g_degree[g_degree==0.0]=1
                norm = torch.pow(g_degree, -0.5).view(-1, 1)
            
                # print(h[src_type])
                # print(etype,G.edges(etype=etype))
                normed=h[self.op_edges[etype]][src_type]
                if etype=='click' or etype=='clicked':
                    
                    
                    G.nodes[src_type].data['normed']=self.dropout(normed)
                else:
                    
                    G.nodes[src_type].data['normed']=normed
                # funcs[etype]=(fn.copy_u('normed', 'm'), fn.mean('m', 'h'))
                G.update_all(message_func=dgl.function.copy_u('normed', 'm'),reduce_func=dgl.function.mean('m','n_f'),etype=(src_type,etype,dst_type))
                
                rst=G.nodes[dst_type].data['n_f']

                
                g_InDegree=G.nodes[dst_type].data[self.op_edges[etype]+'_dg']
                g_InDegree[g_InDegree==0.0]=1
                norm2=g_InDegree/50
                norm2[norm2<1]=1
                norm2=norm2.view(-1,1)
                norm = torch.pow(g_InDegree, -0.5).view(-1, 1)
               
                rst = rst
                if self.op_edges[etype] not in rst_dic:
                    rst_dic[self.op_edges[etype]]={}
                rst_dic[self.op_edges[etype]][dst_type]=rst
            return rst_dic


class GraphLayer(torch.nn.Module):
    def __init__(self,userEmbeddingLayer,itemEmbeddingLayer):
        super(GraphLayer, self).__init__()
        self.userEmbeddingLayer =userEmbeddingLayer
        self.itemEmbeddingLayer =itemEmbeddingLayer
        self.emb={'user':self.userEmbeddingLayer,'item':self.itemEmbeddingLayer}
        self.op_edges={'bought':'buy','buy':'buy','click':'click','clicked':'click','cart':'cart','carted':'cart','fav':'fav','faved':'fav'}
        self.etypes=['cart','click','buy','fav']
        
        self.GCNS=torch.nn.ModuleList(HeterLightGCNLayer(self.op_edges) for i in range(configs['model']['n_gcn_layers']))
        self.n_layers=configs['model']['n_gcn_layers']
        self.g_out_dim=configs['model']['embedding_size']*1
        self.metric=torch.nn.BCELoss()
        self.fc=torch.nn.Linear(self.g_out_dim, self.g_out_dim//2)
       
        self.mulb_user=torch.nn.Sequential(
                                    torch.nn.Linear(self.g_out_dim*4, self.g_out_dim),
                                    torch.nn.ReLU(True),
                                    torch.nn.Linear(self.g_out_dim, self.g_out_dim),
                                    
                                    )

        self.mulb_item=torch.nn.Sequential(
                                    torch.nn.Linear(self.g_out_dim*4, self.g_out_dim),
                                    torch.nn.ReLU(True),
                                    torch.nn.Linear(self.g_out_dim, self.g_out_dim),
                                   
                                    )
        self.fc1=torch.nn.Sequential(torch.nn.Linear(self.g_out_dim, self.g_out_dim//2),
                                    torch.nn.ReLU(True),
                                    torch.nn.Linear(self.g_out_dim//2, self.g_out_dim),
                                    
                                    )
        self.fc2=torch.nn.Sequential(torch.nn.Linear(self.g_out_dim, self.g_out_dim//2),
                                    torch.nn.ReLU(True),
                                    torch.nn.Linear(self.g_out_dim//2, self.g_out_dim),
        )
    def graph2graph(self,graph1,graph2,blocks,user_ids,have_constra):
        graph1_emb=self.fc1(graph1)
        graph2_emb=self.fc1(graph2)
        noconstra_mask=have_constra!=0
        mask=(user_ids.unsqueeze(0)==user_ids.unsqueeze(1))
        mask=~mask
        mask=mask[noconstra_mask]  ## 在这个之前mask是B*B 在喆之后 mask是 have constra num *B
        scores=torch.matmul(graph2_emb,graph1_emb.T)
        
        self_scores=torch.diag(scores).reshape(-1,1)
        scores=scores[noconstra_mask]
        self_scores=self_scores[noconstra_mask]
        scores=(self_scores-scores).sigmoid()
        
        scores=scores[mask]
        
        loss=self.metric(scores,torch.ones_like(scores))
        
        return loss
    def inner_loss(self,buy_emb,click_emb,cart_emb,have_cart):
        buy_emb=self.fc1(buy_emb)
        click_emb=self.fc1(click_emb)
        cart_emb=self.fc1(cart_emb)
        
        mask=have_cart==1
       
        buy_emb=buy_emb[mask]
        click_emb=click_emb[mask]
        cart_emb=cart_emb[mask]
        a=(buy_emb*click_emb).sum(-1)
        b=(buy_emb*cart_emb).sum(-1)
        scores=(a-b).sigmoid()
       
        scores=scores.clamp(max=0.8)
        loss=self.metric(scores,torch.ones_like(scores))
        return loss
    def graph_computer(self,block,h):
        g_cat={etype:{} for etype in self.etypes}
        h1=h
        
        for layer in range(configs['model']['n_gcn_layers']):
            gcn=self.GCNS[layer]
            
            h1=gcn(block,h1)
            ### for evey etype make feature of each ntypes
            for etype,v1 in h1.items():
                # h[k]=F.relu(h[k])
                for ntype,v2 in v1.items():
                    if layer==0:
                        g_cat[etype][ntype]=h1[etype][ntype]
                    else:
                        g_cat[etype][ntype]=g_cat[etype][ntype]+h1[etype][ntype]
        for etype,v1 in h1.items():
               
                for ntype,v2 in v1.items():
                    
                    g_cat[etype][ntype]=g_cat[etype][ntype]/configs['model']['n_gcn_layers']
                    
                    
        return g_cat
    
    def remove_graph_edges(self,block,user_id,pos_items,sampled_click=None,sampled_click_user_id=None):
        new_graphs=[]
        new_graph=block
        src,dst=user_id,pos_items
    
        for etype in [('buy','bought'),('click','clicked'),('cart','carted')]:
            u,i,eidsl=new_graph.edge_ids(src,dst,return_uv=True,etype=etype[0])    
            _,_,eidsr=new_graph.edge_ids(dst,src,return_uv=True,etype=etype[1])
            new_graph=dgl.remove_edges(new_graph,eidsr,etype=etype[1])
            new_graph=dgl.remove_edges(new_graph,eidsl,etype=etype[0])
            new_graph.nodes['user'].data[etype[0]+'_dg'][u]-=1
            new_graph.nodes['item'].data[etype[0]+'_dg'][i]-=1
        
        if sampled_click is not None and configs['model']['remove_click_edges']==1:
            
        
            src,dst=sampled_click_user_id,sampled_click
            
            for etype in [('buy','bought'),('click','clicked'),('cart','carted')]:
                u,i,eidsl=new_graph.edge_ids(src,dst,return_uv=True,etype=etype[0])    
                _,_,eidsr=new_graph.edge_ids(dst,src,return_uv=True,etype=etype[1])
                new_graph=dgl.remove_edges(new_graph,eidsr,etype=etype[1])
                new_graph=dgl.remove_edges(new_graph,eidsl,etype=etype[0])
                new_graph.nodes['user'].data[etype[0]+'_dg'][u]-=1
                new_graph.nodes['item'].data[etype[0]+'_dg'][i]-=1
        
        return new_graph
    def remove_graph_inner_edges(self,block):
        new_graph=block
        dst,src=new_graph.edges(etype='bought')
        src,dst=src.to(configs['train']['device']),dst.to(configs['train']['device'])
        for etype in [('click','clicked')]:
            u,i,eidsl=new_graph.edge_ids(src,dst,return_uv=True,etype=etype[0])    
            _,_,eidsr=new_graph.edge_ids(dst,src,return_uv=True,etype=etype[1])
            new_graph=dgl.remove_edges(new_graph,eidsr,etype=etype[1])
            new_graph=dgl.remove_edges(new_graph,eidsl,etype=etype[0])
            new_graph.nodes['user'].data[etype[0]+'_dg'][u]-=1
            new_graph.nodes['item'].data[etype[0]+'_dg'][i]-=1
        
        return new_graph
    def forward(self,blocks,is_eval=False,constra_b=None,have_cart=None,seq_tensor=None):
        '''
            pos4seq,neg4seq,user4seq is real ID seq of seqModel
        '''
        user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,sampled_view,have_cart,sampled_click,have_constra=seq_tensor
        have_cart=have_cart.squeeze(1)
        have_constra=have_constra.squeeze(1)
        block=blocks
        if not is_eval:
            mask=(pos-neg)!=0
            temp=user_id.repeat(100,1).t()
            userID=temp[mask]
            posItem=pos[mask]
            negItem=neg[mask]
        else:
            posItem=pos.squeeze(1)
            negItem=neg
            userID=user_id
        
        if is_eval:
            sampled_click=sampled_click[sampled_click!=-1]
        else:
            
            sampled_click=sampled_click[sampled_click!=-1]
          
           
        if not configs['model']['no_constra']: # if has constra loss computer it
            ### this is for intra-view CL loss
            constra_userID=userID.unique()
            h={}
            for etype in self.etypes:
                h[etype]={}
                for ntype in ['user','item']:
                    h[etype][ntype]=self.emb[ntype](torch.arange(0,block.num_nodes(ntype),device=configs['train']['device']))
            # constra_graph=self.remove_graph_inner_edges(block) 已经预处理过了不需要再弄了
            constra_graph=blocks
            g_out=self.graph_computer(constra_graph,h)
            idx2b={0:'buy',1:'click',2:'cart'}
            graph_b1=[]
            graph_b2=[]
            for i,j in zip(constra_b,user_id.tolist()):
                b1=idx2b[i[0]]
                b1=g_out[b1]['user'][j]
                graph_b1.append(b1)
                b2=idx2b[i[1]]
                b2=g_out[b2]['user'][j]
                graph_b2.append(b2)
            graph_b1=torch.stack(graph_b1,dim=0)
            graph_b2=torch.stack(graph_b2,dim=0)
            
            # constra_loss=self.graph2graph(g_out['buy']['user'][user4seq],g_out['click']['user'][user4seq],blocks,user4seq)
            constra_loss=self.graph2graph(graph_b1,graph_b2,block,user_id,have_constra)
            inner_loss=self.inner_loss(g_out['buy']['user'][user_id],g_out['click']['user'][user_id],g_out['cart']['user'][user_id],have_cart=have_cart)
            click_graph_emb=(g_out['buy']['user'][user_id],g_out['click']['user'][user_id])
        h={}
        # block=self.remove_graph_edges(blocks,userID,posItem,sampled_click_for_remove_edge,sampled_click_user_id)
        for etype in self.etypes:
            h[etype]={}
            for ntype in ['user','item']:
                h[etype][ntype]=self.emb[ntype](torch.arange(0,block.num_nodes(ntype),device=configs['train']['device']))
        g_out=self.graph_computer(block,h)
        # for batch graph the etype order maybe different so self.etype
        g_cat={}
        for ntype in ['user','item']:
            
            g_cat[ntype]=self.emb[ntype](torch.arange(0,block.num_nodes(ntype),device=configs['train']['device']))
        for etype in ['buy','click','cart']:
            for ntype in g_out[etype]:
                node_emb=g_cat.get(ntype,None)
                if node_emb is None:
                    g_cat[ntype]=g_out[etype][ntype]
                else:
                    g_cat[ntype]=torch.cat((node_emb,g_out[etype][ntype]),dim=1)
        h=g_cat
       
        if not is_eval:
            pos_user_emb=h['user'][userID.squeeze(-1)]
            pos_item_emb=h['item'][posItem.squeeze(-1)]
            neg_item_emb=h['item'][negItem.squeeze(-1)]
            sampled_click_emb=h['item'][sampled_click]
            pos_user_emb=self.mulb_user(pos_user_emb)
            pos_item_emb=self.mulb_item(pos_item_emb)
            neg_item_emb=self.mulb_item(neg_item_emb)
            sampled_click_emb=self.mulb_item(sampled_click_emb)
            return pos_user_emb,pos_item_emb,neg_item_emb,constra_loss,click_graph_emb,inner_loss,sampled_click_emb
        else:

            pos_user_emb=h['user'][userID.squeeze(-1)]
            pos_item_emb=h['item'][posItem.squeeze(-1)]
            neg_item_emb=h['item'][negItem.flatten()]
            sampled_click_emb=h['item'][sampled_click]
         
            pos_user_emb=self.mulb_user(pos_user_emb)
            neg_item_emb=self.mulb_item(neg_item_emb)
            pos_item_emb=self.mulb_item(pos_item_emb)
            sampled_click_emb=self.mulb_item(sampled_click_emb)
            
            return pos_user_emb,pos_item_emb,neg_item_emb,constra_loss,click_graph_emb,inner_loss,sampled_click_emb
