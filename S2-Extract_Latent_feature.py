import os

import pickle

import numpy as np
import pandas as pd

from importlib import reload

import torch
from functools import partial
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from datetime import datetime
from cave_vae_models import *
import shutil
from functools import partial
from tqdm import tqdm
import random
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
def get_data():
    df=pd.read_csv('../ppmi_IQR_BL_name_id_variable.csv')
    print(df.shape)
    patients = df['APPRDX'].values == 1#"PD"
    controls = df['APPRDX'].values == 2#"NC"
    print(df[patients].shape)
    print(df[patients].head())
    # data = np.load('../Data/ppmi_t1_MNI_pd_nc.npy')
    data=np.load("../fast_ori_irq_MNI.npy")
    print(data.shape)
    pd_data = data[patients, :, :, :]
    print(pd_data.shape)
    b=pd_data[0,:,:,:]
    print(b.shape,b.max(),b.min())

    df_mci = pd.read_csv('../ppmi_IQR_BL_name_id_variable_32_mci.csv')
    print(df_mci.shape)
    patients = df_mci['APPRDX'].values == 1  # 1#"PD"
    df_pd_mci = df_mci[patients]
    print(df_pd_mci.shape)
    print(df_pd_mci.head())

    return patients,controls,pd_data,data

def load_VAE(weight_path,model_name):
    batch_size = 64
    if model_name=='vae':
        model = VAE(input_shape=(64,64,64,1),
                                                latent_size=32,
                                                batch_size = batch_size,
                                                disentangle=True,
                                                gamma=100,
                                                kernel_size = 3,
                                                filters = 48,
                                                intermediate_dim = 128,
                                                nlayers = 2,
                                                bias=True)

    state_dict = torch.load(weight_path, map_location='cpu')
    load_weights_dict = {k: v for k, v in state_dict.items()
                         if "fc" not in k and model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    return model
def load_CVAE(weight_path,model_name,gamma_model):
    latent_dim = 16
    batch_size = 32
    beta = 1
    gamma = gamma_model#100
    disentangle = True
    if model_name == 'cave':
        model = CVAE(latent_size=latent_dim, beta=beta, disentangle=disentangle, gamma=gamma, bias=True,
                        batch_size=batch_size)
    state_dict = torch.load(weight_path, map_location='cpu')
    load_weights_dict = {k: v for k, v in state_dict.items()
                         if "fc" not in k and model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    return model
def get_latent(data,n_samples,cvae,vae,save_path,save_name):
    data=np.expand_dims(data,axis=1)#330 1 64 64 64
    data=torch.FloatTensor(data)
    salient_vec_abide=[]
    background_vec_abide=[]
    vae_vec_abide=[]
    print(data[:,:,:,:].shape)
    cvae.eval()
    vae.eval()
    with torch.no_grad():
        for i in range(n_samples):
            print(i)
            s_encoder=cvae.encoder_S(data)[2]#[:,:,:,:]
            # print("s_encoder", s_encoder.shape)
            z_encoder=cvae.encoder_Z(data)[2]
            # print("z_encoder", z_encoder.shape)
            vae_out=vae.encoder(data)[2]
            # print("vae_out",vae_out.shape)

            s_encoder= s_encoder.detach().numpy()
            z_encoder = z_encoder.detach().numpy()
            vae_out=vae_out.detach().numpy()

            salient_vec_abide.append(s_encoder)
            background_vec_abide.append(z_encoder)
            vae_vec_abide.append(vae_out)

    salient_vec_abide=np.array(salient_vec_abide)
    background_vec_abide=np.array(background_vec_abide)
    vae_vec_abide=np.array(vae_vec_abide)
    # salient_vec_abide = np.array([cvae.encoder_S(data[:,:,:,:])[2] for _ in range(n_samples)])
    # background_vec_abide = np.array([cvae.encoder_Z(data[:,:,:,:])[2] for _ in range(n_samples)])
    # vae_vec_abide = np.array([vae(data[:,:,:,:])[2] for _ in range(n_samples)])


    fn = os.path.join(save_path,'latent_vecs{}_{}.npz'.format(n_samples,save_name))
    np.savez_compressed(fn,
                        salient_vec_abide=salient_vec_abide,
                        background_vec_abide=background_vec_abide,
                        vae_vec_abide=vae_vec_abide)
if __name__ == '__main__':
    setup_seed(2)
    vae_weight_path='../vae_bspline_v2_100_0.9_0.001.pth'
    cvae_weight_path = '../cvae_bspline_10_1_0.9_0.001.pth'
    cvae_choice = ['cave']
    vae_choice=['vae']
    save_name=cvae_weight_path.split('/')[-1].split('.pth')[0]
    save_path="./step2_result"#_MNI_iqr
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    patients,controls,pd_data,data=get_data()
    vae=load_VAE(vae_weight_path,vae_choice[1])
    cvae=load_CVAE(cvae_weight_path,cvae_choice[0],gamma_model=10)
    get_latent(data, 10, cvae, vae, save_path, save_name)
    get_latent(data,100,cvae,vae,save_path,save_name)





