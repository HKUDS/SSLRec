import torch 
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class CML(BaseModel):
    def __init__(self, data_handler):
        super(CML, self).__init__(data_handler)

        self.userNum = data_handler.userNum
        self.itemNum = data_handler.itemNum
        self.behavior = data_handler.behaviors
        self.behavior_mats = data_handler.behavior_mats
        
        self.embedding_dict = self._init_embedding() 
        self.weight_dict = self._init_weight()
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)

    def _init_embedding(self):
        
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def _init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([configs['model']['hidden_dim'], configs['model']['hidden_dim']]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([configs['model']['hidden_dim'], configs['model']['hidden_dim']]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([configs['model']['head_num']*configs['model']['hidden_dim'], configs['model']['hidden_dim']]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict  


    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def forward(self):
        user_embed, item_embed, user_embeds, item_embeds = self.gcn()
        return user_embed, item_embed, user_embeds, item_embeds 

    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
        loss = bpr_loss + self.reg_weight * reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    def para_dict_to_tenser(self, para_dict): 
    
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = configs['model']['hidden_dim']

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(configs['model']['drop_rate']) 

        self.gnn_layer = configs['model']['gnn_layer'] 
        self.layers = nn.ModuleList()
        for i in range(0, self.gnn_layer):  
            self.layers.append(GCNLayer(configs['model']['hidden_dim'], configs['model']['hidden_dim'], self.userNum, self.itemNum, self.behavior, self.behavior_mats))  

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, configs['model']['hidden_dim'])
        item_embedding = torch.nn.Embedding(self.itemNum, configs['model']['hidden_dim'])
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['gnn_layer']*configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        u_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['gnn_layer']*configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        i_input_w = nn.Parameter(torch.Tensor(configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        u_input_w = nn.Parameter(torch.Tensor(configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        nn.init.xavier_uniform_(i_concatenation_w)
        nn.init.xavier_uniform_(u_concatenation_w)
        nn.init.xavier_uniform_(i_input_w)
        nn.init.xavier_uniform_(u_input_w)
        # init.xavier_uniform_(alpha)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        for i, layer in enumerate(self.layers):
            
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)
            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)  

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
            
        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w)
            

        return user_embedding, item_embedding, user_embeddings, item_embeddings  #[31882, 16], [31882, 16], [4, 31882, 16], [4, 31882, 16]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.userNum = userNum
        self.itemNum = itemNum
        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.i_w)
        nn.init.xavier_uniform_(self.u_w)

    def forward(self, user_embedding, item_embedding):
        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)
        for i in range(len(self.behavior)):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding)
        user_embeddings = torch.stack(user_embedding_list, dim=0) 
        item_embeddings = torch.stack(item_embedding_list, dim=0)
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))
        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))
        return user_embedding, item_embedding, user_embeddings, item_embeddings             





class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param
    def named_leaves(self):
        return []
    def named_submodules(self):
        return []
    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class MetaWeightNet(nn.Module):
    def __init__(self, beh_num):
        super(MetaWeightNet, self).__init__()
        self.beh_num = beh_num
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.LeakyReLU(negative_slope=configs['model']['slope'])  
        self.prelu = torch.nn.PReLU()
        self.relu = torch.nn.ReLU()
        self.tanhshrink = torch.nn.Tanhshrink()
        self.dropout7 = torch.nn.Dropout(configs['model']['drop_rate'])
        self.dropout5 = torch.nn.Dropout(configs['model']['drop_rate1'])
        self.batch_norm = torch.nn.BatchNorm1d(1)
        initializer = nn.init.xavier_uniform_
        self.SSL_layer1 = nn.Linear(configs['model']['hidden_dim']*3, int((configs['model']['hidden_dim']*3)/2))
        self.SSL_layer2 = nn.Linear(int((configs['model']['hidden_dim']*3)/2), 1)
        self.SSL_layer3 = nn.Linear(configs['model']['hidden_dim']*2, 1)
        self.RS_layer1 = nn.Linear(configs['model']['hidden_dim']*3, int((configs['model']['hidden_dim']*3)/2))
        self.RS_layer2 = nn.Linear(int((configs['model']['hidden_dim']*3)/2), 1)
        self.RS_layer3 = nn.Linear(configs['model']['hidden_dim'], 1)
        self.beh_embedding = nn.Parameter(initializer(torch.empty([beh_num, configs['model']['hidden_dim']]))).cuda()
 
    def forward(self, infoNCELoss_list, behavior_loss_multi_list, user_step_index, user_index_list, user_embeds, user_embed):  
        
        infoNCELoss_list_weights = [None]*self.beh_num
        behavior_loss_multi_list_weights = [None]*self.beh_num
        for i in range(self.beh_num):
            SSL_input = configs['model']['inner_product_mult']*torch.cat((configs['model']['inner_product_mult']*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, configs['model']['hidden_dim'])*configs['model']['inner_product_mult'], user_embeds[i][user_step_index]), 1), user_embed[user_step_index]), 1)
            SSL_input3 = configs['model']['inner_product_mult']*((infoNCELoss_list[i].unsqueeze(1).repeat(1, configs['model']['hidden_dim']*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))
            infoNCELoss_list_weights[i] = configs['model']['inner_product_mult']*self.sigmoid(self.batch_norm(np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))).squeeze()).unsqueeze(1)).squeeze())
            SSL_weight3 = configs['model']['inner_product_mult']*self.sigmoid(self.batch_norm(self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))).squeeze())
            infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2
            RS_input = configs['model']['inner_product_mult']*torch.cat((configs['model']['inner_product_mult']*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, configs['model']['hidden_dim'])*configs['model']['inner_product_mult'], user_embed[user_index_list[i]]), 1), user_embeds[i][user_index_list[i]]), 1)
            RS_input3 = configs['model']['inner_product_mult']*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, configs['model']['hidden_dim']))*user_embed[user_index_list[i]])
            behavior_loss_multi_list_weights[i] = configs['model']['inner_product_mult']*self.sigmoid(self.batch_norm(np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(self.dropout7(self.prelu(self.RS_layer1(RS_input)))).squeeze()).unsqueeze(1))).squeeze()
            RS_weight3 = configs['model']['inner_product_mult']*self.sigmoid(self.batch_norm(self.dropout7(self.prelu(self.RS_layer3(RS_input3)))).squeeze()).squeeze()
            behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
        return infoNCELoss_list_weights, behavior_loss_multi_list_weights



