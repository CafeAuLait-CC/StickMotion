import torch
from torch import nn
from einops import rearrange, repeat, reduce
from copy import deepcopy

class StickmanEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_in = nn.Linear(cfg.in_dim, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, 
                                                   dropout=cfg.dropout,
                                                   activation=cfg.activation,
                                                   dim_feedforward=cfg.ff_dim,
                                                   batch_first=True,
                                                   nhead=cfg.nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.proj_out = nn.Linear(6*cfg.d_model, cfg.out_dim)
        
    def forward(self, x): #[b, 6, 64, 2] should be track for stickman
        x = rearrange(x, 'b n p c -> b n (p c)')
        x = self.proj_in(x)
        x = self.encoder(x) # [b, 6, 128]
        x = rearrange(x, 'b n e -> b (n e)')
        x = self.proj_out(x)
        return x

class FcnBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(FcnBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        residual = x
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x += residual
        return x

class FCN(nn.Module):
    def __init__(self, in_dim, out_dim, dim_list, dropout=0.1):
        super().__init__()
        dim_list = deepcopy(dim_list)
        dim_list.insert(0, in_dim)
        fcn_blocks = [FcnBlock(dim_list[i], dim_list[i+1], dropout) for i in range(len(dim_list)-1)]
        self.fcn_blocks = nn.ModuleList(fcn_blocks)
        self.proj_out = nn.Linear(dim_list[-1], out_dim)
        self.dim_list = dim_list
        
        
    def forward(self, x): #[b, 251] should be original pose
        start_dim = self.dim_list[0]
        for dim, fcn_block in zip(self.dim_list[1:], self.fcn_blocks):
            residual = x
            x = fcn_block(x)
            if dim == start_dim:
                start_dim = dim
                x += residual
        x = self.proj_out(x)
        return x

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fcn = FCN(cfg.in_dim, cfg.out_dim, cfg.fcn_dims, cfg.dropout)
        
        
    def forward(self, x): #[b, 251] should be original pose
        x = self.fcn(x)
        return x

class StickmanDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fcn = FCN(cfg.in_dim, cfg.out_dim*cfg.candidate_num, cfg.fcn_dims, cfg.dropout)
        self.candidate_num = cfg.candidate_num
    
    def forward(self, x): #[b, feat] should be feat
        x = self.fcn(x)
        x = rearrange(x, 'b (d n c) -> b d n c', d=self.candidate_num, c=3) # [b, 22, 3]
        return x
    
    

class StickmanDecoder1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.candidate_num = cfg.candidate_num
        self.fcn = FCN(cfg.in_dim, cfg.in_dim*self.candidate_num, cfg.fcn_dims, cfg.dropout)
        self.fcn_out = nn.ModuleList([FCN(cfg.in_dim, cfg.out_dim, cfg.bran_dims, cfg.dropout) for _ in range(self.candidate_num)])
    
    def forward(self, x): #[b, feat] should be feat
        x = self.fcn(x)
        x = rearrange(x, 'b (d e) -> b d e', d=self.candidate_num)
        _x = []
        for i in range(self.candidate_num):
            _x.append(self.fcn_out[i](x[:,i])[:,None])
        x = torch.cat(_x, dim=1)
        x = rearrange(x, 'b d (n c) -> b d n c', c=3) # [b, 22, 3]
        return x
    
    
    
# class StickmanDecoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.hou = 4
#         self.fcn = FCN(cfg.in_dim, cfg.out_dim*self.hou, cfg.fcn_dims, cfg.dropout)
        
        
#     def forward(self, x): #[b, feat] should be feat
#         x = self.fcn(x)
#         x = rearrange(x, 'b (h n c) -> b h n c', h=self.hou, c=3) # [b, 22, 3]
#         return x