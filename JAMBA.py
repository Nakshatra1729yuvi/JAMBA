import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum,rearrange,repeat



torch.set_default_dtype(torch.float16)



class HybridConfig:
  def __init__(self):
    self.vocab_size=50257
    self.n_head=8
    self.n_embd=1184
    self.block_size=16192
    self.dropout=0.1
    self.n_intmd=4048
    self.ssm_hid=64
    self.kernel=4
    self.num_layers=8
    self.ssm_delta=64
    self.n_exp=16
    self.k=2






class MHA(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    assert config.n_embd%config.n_head==0
    self.n_embd=config.n_embd
    self.n_head=config.n_head
    self.attn_proj=nn.Linear(config.n_embd,3*config.n_embd)
    self.c_proj=nn.Linear(config.n_embd,config.n_embd)
    self.d_k=self.n_embd//self.n_head
    self.register_buffer('bias',torch.tril(torch.ones((1,1,config.block_size,config.block_size))))
    self.attn_dropout=nn.Dropout(config.dropout)
    self.resid_dropout=nn.Dropout(config.dropout)
  def forward(self,x):
    B,T,C=x.shape
    q,k,v=self.attn_proj(x).split(self.n_embd,dim=-1)
    q=q.view(B,T,self.n_head,self.d_k).transpose(1,2)
    k=k.view(B,T,self.n_head,self.d_k).transpose(1,2)
    v=v.view(B,T,self.n_head,self.d_k).transpose(1,2)
    attn=(q@k.transpose(-1,-2))*(1/(math.sqrt(self.d_k)))
    attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
    attn=F.softmax(attn,dim=-1)
    attn=self.attn_dropout(attn)

    y=attn@v
    y=y.transpose(1,2).contiguous().view(B,T,C)
    return self.resid_dropout(self.c_proj(y))







class Mamba(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.ssm_hid=config.ssm_hid
    self.intermediate_size=config.n_intmd
    self.delta_rank=config.ssm_delta

    self.in_proj=nn.Linear(config.n_embd,config.n_intmd*2)

    self.conv=nn.Conv1d(in_channels=config.n_intmd,out_channels=config.n_intmd,kernel_size=config.kernel,padding=config.kernel-1,groups=config.n_intmd)

    self.x_proj=nn.Linear(config.n_intmd,config.ssm_delta+2*config.ssm_hid)
    self.delta_proj=nn.Linear(config.ssm_delta,config.n_intmd)

    self.A=repeat((torch.arange(config.ssm_hid)),'n -> d n',d=config.n_intmd)
    self.A_log=nn.Parameter(torch.log(self.A))
    self.D=nn.Parameter(torch.ones(config.n_intmd))

    self.out_proj=nn.Linear(config.n_intmd,config.n_embd)

  def forward(self,x):
    batch_size,seq_len,_=x.shape
    xz=self.in_proj(x)
    xz=rearrange(xz,"b l x -> b x l")
    x,z=xz.chunk(2,dim=1)

    x=self.conv(x)[:,:,:seq_len]
    x=F.silu(x)

    y=self.ssm(x)

    y=y*F.silu(z)

    output=self.out_proj(rearrange(y,"b d l -> b l d"))

    return output
    print(output.shape,z.shape)

  def ssm(self,x):

    A=-torch.exp(self.A_log.float())
    D=self.D.float()

    x_rearrange=rearrange(x,'b d l -> b l d')
    x_rearrange=self.x_proj(x_rearrange)
    delta,B,C=x_rearrange.split([self.delta_rank,self.ssm_hid,self.ssm_hid],dim=-1)
    delta=F.softplus(self.delta_proj(delta))

    y=self.selective_scan(x,A,B,C,D,delta)

    return y

  def selective_scan(self,u,A,B,C,D,delta):
    b,d_in,l=u.shape
    n=A.shape[1]
    deltaA=torch.exp(einsum(delta,A,'b l d_in , d_in n -> b d_in l n'))
    deltaB_u=einsum(delta,B,u,'b l d_in , b l n , b d_in l -> b d_in l n')
    x=torch.zeros((b,d_in,n),device=next(self.parameters()).device)
    ys=[]
    for i in range(l):
      x=deltaA[:,:,i]*x+deltaB_u[:,:,i]
      y=einsum(x,C[:,i,:],'b d_in n , b n -> b d_in')
      ys.append(y)
    y=torch.stack(ys,dim=2)

    y=y+u*rearrange(D,'d_in -> d_in 1')

    return y






class MLP(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.up_proj=nn.Linear(config.n_embd,config.n_embd*4)
    self.gate_proj=nn.Linear(config.n_embd,config.n_embd*4)
    self.down_proj=nn.Linear(config.n_embd*4,config.n_embd)
  def forward(self,x):
    return self.down_proj(self.up_proj(x)*F.silu(self.gate_proj(x)))





class MOE(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.n_experts=config.n_exp
    self.k=config.k
    self.gate_proj=nn.Linear(config.n_embd,config.n_exp)
    self.experts=nn.ModuleList([MLP(config) for _ in range(config.n_exp)])
  def forward(self,x):
    B,T,C=x.shape
    x_flat=x.reshape(-1,C)
    gate_score=self.gate_proj(x_flat)
    outputs=torch.zeros_like(x_flat)
    top_val,top_idx=torch.topk(gate_score,self.k)
    for i in range(self.k):
      expert_idx=top_idx[:,i]
      expert_val=top_val[:,i]
      for e in range(self.n_experts):
        mask=(expert_idx==e)
        if mask.sum()==0:
          continue
        out=self.experts[e](x_flat[mask])
        outputs[mask]+=out*expert_val[mask].unsqueeze(-1)
    return outputs.reshape(B,T,C)







class Attn_Block(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.rn1=nn.RMSNorm(config.n_embd)
    self.rn2=nn.RMSNorm(config.n_embd)
    self.attn=MHA(config)
    self.ff=MLP(config)
  def forward(self,x):
    x=x+self.attn(self.rn1(x))
    x=x+self.ff(self.rn2(x))
    return x





class Attn_MOE_Block(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.rn1=nn.RMSNorm(config.n_embd)
    self.rn2=nn.RMSNorm(config.n_embd)
    self.attn=MHA(config)
    self.ff=MOE(config)
  def forward(self,x):
    x=x+self.attn(self.rn1(x))
    x=x+self.ff(self.rn2(x))
    return x







class Mamba_Block(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.rn1=nn.RMSNorm(config.n_embd)
    self.rn2=nn.RMSNorm(config.n_embd)
    self.mamba=Mamba(config)
    self.ff=MLP(config)
  def forward(self,x):
    x=x+self.mamba(self.rn1(x))
    x=x+self.ff(self.rn2(x))
    return x








class Mamba_MOE_Block(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.rn1=nn.RMSNorm(config.n_embd)
    self.rn2=nn.RMSNorm(config.n_embd)
    self.mamba=Mamba(config)
    self.ff=MOE(config)
  def forward(self,x):
    x=x+self.mamba(self.rn1(x))
    x=x+self.ff(self.rn2(x))
    return x






eot_token=50257









class HybridBlock(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.net=nn.Sequential(
        Mamba_Block(config),
        Mamba_MOE_Block(config),
        Mamba_Block(config),
        Mamba_MOE_Block(config),
        Attn_Block(config),
        Mamba_MOE_Block(config),
        Mamba_Block(config),
        Mamba_MOE_Block(config)
    )
  def forward(self,x):
    return self.net(x)







class Jamba(nn.Module):
  def __init__(self,config:HybridConfig):
    super().__init__()
    self.block_size=config.block_size
    self.token_emb=nn.Embedding(config.vocab_size,config.n_embd)
    self.pos_emb=nn.Embedding(config.block_size,config.n_embd)
    self.drop=nn.Dropout(config.dropout)
    self.blocks=nn.ModuleList([HybridBlock(config) for _ in range(config.num_layers)])
    self.rn=nn.RMSNorm(config.n_embd)
    self.head=nn.Linear(config.n_embd,config.vocab_size)
  def forward(self,ids,labels=None):
    B,T=ids.shape
    assert T<=self.block_size
    pos=torch.arange(0,T,dtype=torch.long,device=ids.device).unsqueeze(0)
    emb_ids=self.token_emb(ids)+self.pos_emb(pos)
    x=self.drop(emb_ids)

    for blk in self.blocks:
      x=blk(x)

    x=self.rn(x)
    logits=self.head(x)
    loss=None
    if labels is not None:
      loss=F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1))
    return logits,loss
  @torch.no_grad()
  def generate(self,idx,max_tokens=50,temprature=0.8,topk=None):
    for _ in range(max_tokens):
      idx=idx[:,-self.block_size:]
      logits,_=self(idx)
      logit=logits[:,-1,:]/temprature
      if topk is not None:
        v,i=torch.topk(logit,topk)
        mask=logit<v[:,-1].unsqueeze(-1)
        mask=mask.to(logit.device)
        logit[mask]=float("-inf")
      prob=F.softmax(logit,dim=-1)
      next_token=torch.multinomial(prob,num_samples=1)
      idx=torch.concat([idx,next_token],dim=-1)
      if next_token.item()==eot_token:
        break
    return idx






print("Creating Model...")



c=HybridConfig()
model=Jamba(c)



def count_parameters(model):
  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total Trainable Parameters:{trainable_parameters:,}")

count_parameters(model)

# print(model.generate(torch.randint(0,50000,[1,5])))



#####################################################################
# Output:
# Creating Model...
# Total Trainable Parameters:10,220,852,913