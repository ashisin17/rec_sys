"""
LightGCN recommender for Sim4Rec
Ashita Singh
"""

from __future__ import annotations
import numpy as np, pandas as pd, torch, random
import torch.nn as nn, torch.nn.functional as F
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, DataFrame

# ───────── reproducibility
def _seed_all(seed:int=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
_seed_all(42)

# ───────── helpers
def degree(idx:torch.Tensor,dtype=torch.float32):
    n=int(idx.max())+1 if idx.numel() else 0
    return torch.bincount(idx,minlength=n).to(dtype)

def _dcg_rev(rank_idx:torch.Tensor,price:torch.Tensor,gt:torch.Tensor)->float:
    b,k=rank_idx.shape
    disc=1/torch.log2(torch.arange(k,device=rank_idx.device)+2)
    hit=rank_idx.eq(gt.unsqueeze(1))
    any_hit=hit.any(1)
    if not any_hit.any(): return 0.0
    pos=hit.float().argmax(1)
    rev=torch.zeros(b,device=rank_idx.device)
    rev[any_hit]=price[gt[any_hit]]*disc[pos[any_hit]]
    return rev.mean().item()

# ───────── backbone
class _LightGCN(nn.Module):
    def __init__(self,n_u:int,n_i:int,dim:int=128,n_layers:int=2):
        super().__init__()
        self.n_u,self.n_i,self.n_layers=n_u,n_i,n_layers
        self.u_emb=nn.Embedding(n_u,dim); self.i_emb=nn.Embedding(n_i,dim)
        nn.init.xavier_uniform_(self.u_emb.weight); nn.init.xavier_uniform_(self.i_emb.weight)
        self.edge=None
    @torch.no_grad()
    def set_edges(self,e:np.ndarray,dev:str):
        u=torch.from_numpy(e[:,0]).long()
        i=torch.from_numpy(e[:,1]).long()+self.n_u
        self.edge=torch.stack([torch.cat([u,i]),torch.cat([i,u])]).to(dev)
    def _prop(self,x,edge):
        r,c=edge
        deg=degree(torch.cat([r,c]),x.dtype)
        norm=(deg[r].clamp(1e-10).pow(-.5)*deg[c].clamp(1e-10).pow(-.5)).unsqueeze(1)
        out=torch.zeros_like(x); out.index_add_(0,r,x[c]*norm); return out
    def forward(self,edge):
        all_e=torch.cat([self.u_emb.weight,self.i_emb.weight]); outs=[all_e]
        for _ in range(self.n_layers):
            all_e=self._prop(all_e,edge); outs.append(all_e)
        fin=torch.stack(outs).mean(0)
        return fin[:self.n_u],fin[self.n_u:]

# ───────── wrapper
class LightGCNRecommender:
    def __init__(self,spark:SparkSession,*,
                 n_epochs:int=30,dim:int=128,n_layers:int=2,
                 lr:float=8e-4,batch_size:int=4096,
                 reg_lambda:float=8e-4,edge_dropout:float=0.50,
                 patience:int=8,seed:int=42,
                 price_exp:float=0.60,root_price_loss:bool=True,
                 device:str="auto"):
        self.spark=spark
        self.cfg=dict(n_epochs=n_epochs,dim=dim,n_layers=n_layers,lr=lr,
                      batch_size=batch_size,reg_lambda=reg_lambda,
                      edge_dropout=edge_dropout,patience=patience,
                      price_exp=price_exp,root_price_loss=root_price_loss)
        _seed_all(seed)
        if device=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"
            try: torch.zeros(1,device=device)
            except: device="cpu"
        self.dev=device
        self.model=None; self.item_price=None; self._w=None

    # light refresh for simulator
    @staticmethod
    def _retrain_cfg(): return dict(epochs=3,lr=4e-4,edge_dropout=0.0,reg_lambda=8e-4)

    # ─── fit
    def fit(self,log:DataFrame,*,user_features=None,item_features=None,**ov):
        cfg={**self.cfg,**ov}
        self.n_u=int((user_features or log).agg(sf.max("user_idx")).first()[0])+1
        self.n_i=int((item_features or log).agg(sf.max("item_idx")).first()[0])+1
        edges=log.select("user_idx","item_idx").distinct().toPandas().astype(np.int64).values
        if edges.size==0: raise ValueError("empty log")

        # price
        if item_features is not None and "price" in item_features.columns:
            price_arr=(item_features.select("item_idx","price").toPandas()
                       .sort_values("item_idx")["price"]
                       .reindex(range(self.n_i),
                                fill_value=item_features.agg(sf.avg("price")).first()[0])
                       .values.astype(np.float32))
        else: price_arr=np.ones(self.n_i,dtype=np.float32)
        p=torch.log1p(torch.from_numpy(price_arr)); p=p/p.mean()
        self.item_price=p.to(self.dev)
        self._w=torch.sqrt(self.item_price) if cfg["root_price_loss"] else self.item_price

        # model & opt
        self.model=_LightGCN(self.n_u,self.n_i,cfg["dim"],cfg["n_layers"]).to(self.dev)
        self.model.set_edges(edges,self.dev)
        opt=torch.optim.AdamW(self.model.parameters(),lr=cfg["lr"],weight_decay=5e-3)

        u=torch.from_numpy(edges[:,0]).to(self.dev)
        i=torch.from_numpy(edges[:,1]).to(self.dev)
        m=u.size(0); val_mask=torch.rand(m,device=self.dev)<.10
        u_tr,u_val=u[~val_mask],u[val_mask]; i_tr,i_val=i[~val_mask],i[val_mask]; m_tr=u_tr.size(0)

        best_val,bad,best_state=-1e9,0,None
        for ep in range(cfg["n_epochs"]):
            perm=torch.randperm(m_tr,device=self.dev); u_tr,i_tr=u_tr[perm],i_tr[perm]
            edge_cur=self.model.edge[:,torch.rand(self.model.edge.size(1),device=self.dev)>=cfg["edge_dropout"]]

            self.model.train(); tot=0.0
            for a in range(0,m_tr,cfg["batch_size"]):
                b=min(a+cfg["batch_size"],m_tr)
                u_b,i_b=u_tr[a:b],i_tr[a:b]
                j_b=torch.randint(0,self.n_i,(b-a,),device=self.dev)  # uniform negatives
                opt.zero_grad()
                ue,ie=self.model(edge_cur)
                pos=(ue[u_b]*ie[i_b]).sum(1); neg=(ue[u_b]*ie[j_b]).sum(1)
                bpr=(self._w[i_b]*-F.logsigmoid(pos-neg)).mean()
                reg=(ue[u_b].norm(2)**2+ie[i_b].norm(2)**2+ie[j_b].norm(2)**2)/2
                loss=bpr+cfg["reg_lambda"]*reg/(b-a); loss.backward(); opt.step(); tot+=loss.item()*(b-a)
            # emb norm
            with torch.no_grad():
                self.model.u_emb.weight[:]=F.normalize(self.model.u_emb.weight,dim=1)
                self.model.i_emb.weight[:]=F.normalize(self.model.i_emb.weight,dim=1)

            # val
            self.model.eval()
            with torch.no_grad():
                ue_f,ie_f=self.model(edge_cur)
                score=torch.sigmoid(ue_f[u_val]@ie_f.T)*(self.item_price**cfg["price_exp"])
                val=_dcg_rev(torch.topk(score,20,dim=1).indices,self.item_price,i_val)
            print(f"[v3.3] ep {ep+1:02d}/{cfg['n_epochs']} loss {tot/m_tr:.4f} val_drev {val:.5f}")
            if val>best_val+1e-5:
                best_val,bad,val_state=val,0,{k:v.clone() for k,v in self.model.state_dict().items()}
            else: bad+=1
            if bad>=cfg["patience"]: print("⏹ early stop"); break
        self.model.load_state_dict(val_state); self.model.eval()

    # ─── predict
    def predict(self,log:DataFrame,k:int,*,users:DataFrame,items:DataFrame,
                user_features=None,item_features=None,filter_seen_items=True,**_)->DataFrame:
        if self.model is None: raise RuntimeError("fit first")
        uid=[u for (u,) in users.select("user_idx").collect() if u<self.n_u]
        iid=[i for (i,) in items.select("item_idx").collect() if i<self.n_i]
        with torch.no_grad():
            ue,ie=self.model(self.model.edge)
            sc=torch.sigmoid(ue[uid]@ie[iid].T)*(self.item_price[iid]**self.cfg["price_exp"])
            sc=sc.cpu().numpy()
        if filter_seen_items and log is not None:
            seen=log.select("user_idx","item_idx").where(sf.col("user_idx").isin(uid)).toPandas()
            for u,i in seen.values:
                if u in uid and i in iid: sc[uid.index(u),iid.index(i)]=-np.inf
        top=np.argpartition(-sc,k-1,axis=1)[:,:k]
        rows=[]
        for r,u in enumerate(uid):
            idxs,vals=top[r],sc[r,top[r]]
            for j in np.argsort(-vals):
                rows.append((u,iid[int(idxs[j])],float(vals[j])))
        return self.spark.createDataFrame(
            self.spark.sparkContext.parallelize(rows),
            schema=["user_idx","item_idx","relevance"])
