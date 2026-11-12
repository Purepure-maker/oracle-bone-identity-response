"""Whole-image residual descriptor from appearance-conditioned spatial relations."""
import copy
import hashlib
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def coordinate_permutations(keys,n,device):
    rows=[]
    for key in keys:
        seed=int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8],'little')
        rows.append(np.random.default_rng(seed).permutation(n))
    return torch.as_tensor(np.asarray(rows),device=device,dtype=torch.long)

def select_tokens(student_map,teacher_map,k,keys,mode):
    b,d,h,w=student_map.shape
    if teacher_map.shape!=student_map.shape or h*w<k:raise ValueError('incompatible token maps')
    energy=teacher_map.detach().abs().sum(1).flatten(1)
    ids=torch.argsort(energy,dim=1,descending=True,stable=True)[:,:k]
    tokens=student_map.flatten(2).transpose(1,2).gather(1,ids[:,:,None].expand(-1,-1,d))
    ys,xs=torch.meshgrid(torch.linspace(-1,1,h,device=student_map.device),torch.linspace(-1,1,w,device=student_map.device),indexing='ij')
    grid=torch.stack([xs.flatten(),ys.flatten()],1)
    xy=grid[ids];weights=energy.gather(1,ids).clamp_min(1e-8)
    weights=weights/weights.sum(1,keepdim=True)
    centered=xy-(xy*weights[:,:,None]).sum(1,keepdim=True)
    radius=((centered.square().sum(-1)*weights).sum(1,keepdim=True)).sqrt().clamp_min(.1)
    xy=centered/radius[:,:,None]
    if mode=='F0':xy=xy*0
    elif mode=='F2':xy=xy.gather(1,coordinate_permutations(keys,k,xy.device)[:,:,None].expand(-1,-1,2))
    elif mode!='F1':raise ValueError('unknown arm')
    return tokens,xy,ids

class RelationHead(nn.Module):
    def __init__(self,input_dim=2048,hidden=64,output_dim=128):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.edge=nn.Sequential(nn.Linear(hidden*2+6,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.out=nn.Sequential(nn.Linear(hidden*2,output_dim),nn.LayerNorm(output_dim))

    def forward(self,tokens,xy):
        h=self.node(tokens);b,k,d=h.shape
        hi=h[:,:,None,:].expand(-1,-1,k,-1);hj=h[:,None,:,:].expand(-1,k,-1,-1)
        delta=xy[:,:,None,:]-xy[:,None,:,:]
        dx,dy=delta.unbind(-1)
        geo=torch.stack([dx,dy,dx.square(),dy.square(),dx*dy,(dx.square()+dy.square()+1e-8).sqrt()],-1)
        # All directed edges; omit self edges. No image pair or gallery interaction here.
        edges=self.edge(torch.cat([hi,hj,geo],-1))
        mask=~torch.eye(k,dtype=torch.bool,device=h.device)
        pooled=edges[:,mask].mean(1)
        value=self.out(torch.cat([h.mean(1),pooled],-1))
        return F.normalize(value,dim=-1)

def fuse(global_z,local_z,alpha):
    if not 0<=alpha<=1:raise ValueError('fusion alpha outside [0,1]')
    g=F.normalize(global_z,dim=-1);l=F.normalize(local_z,dim=-1)
    return torch.cat([math.sqrt(1-alpha)*g,math.sqrt(alpha)*l],-1)

def ranking_loss(z,temperature=.05,margin=.05):
    if len(z)%8:raise ValueError('whole eight-image tuples required')
    v=z.reshape(-1,8,z.shape[-1]);positive=(v[:,0]*v[:,1]).sum(-1)
    negative=(v[:,0,None]*v[:,2:]).sum(-1)
    loss=F.softplus((negative-positive[:,None]+margin)/temperature).mean()
    return loss,{'positive_cosine':float(positive.detach().mean()),'negative_cosine':float(negative.detach().mean()),
        'negative_over_positive_fraction':float((negative>positive[:,None]).float().mean())}

class FusionModel(nn.Module):
    def __init__(self,teacher,c,arm):
        super().__init__();self.teacher=teacher.eval().requires_grad_(False)
        self.local_s4=copy.deepcopy(teacher.globalmodel.s4)
        self.local_s4.requires_grad_(True)
        self.head=RelationHead(c.get('map_dim',2048),c['token_dim'],c['local_dim'])
        self.c=c;self.arm=arm
        self.train(True)

    def train(self,mode=True):
        super().train(mode);self.teacher.eval()
        for m in self.local_s4.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval().requires_grad_(False)
        return self

    def forward(self,x,keys):
        with torch.no_grad():
            f3,f4=self.teacher.globalmodel(x)
            g=self.teacher.bn(self.teacher.fc(self.teacher.pool_g(f4).flatten(1)))
            g=F.normalize(g,dim=-1)
        local=self.local_s4(f3.detach())
        tokens,xy,_=select_tokens(local,f4,self.c['top_tokens'],keys,self.arm)
        z=self.head(tokens,xy)
        return fuse(g,z,self.c['fusion_alpha']),g,z

    def branch_state(self):return {'local_s4':self.local_s4.state_dict(),'head':self.head.state_dict()}
    def load_branch(self,state):
        self.local_s4.load_state_dict(state['local_s4'],strict=True);self.head.load_state_dict(state['head'],strict=True)
