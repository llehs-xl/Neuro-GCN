import numpy as np
import torch


def compute_A(time_series3):
    time_series1=time_series3.squeeze()
    time_series1=time_series1[:,0,:,:]
    c=time_series1[0]
    for i in range(1,len(time_series3)):
        c=np.r_[c,time_series1[i]]
    this_A=np.corrcoef(c.T)
    return this_A


def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat((x_train,x_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return x_train,y_train,x_valid,y_valid


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod