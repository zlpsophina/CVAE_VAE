import shutil
import ants
import os
import numpy as np
# from matplotlib import pyplot as plt
# import umap
from IPython import display
import time
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
# import seaborn as sns
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch
import scipy
import nibabel as nib
import vtk
# from nibabel.viewers import OrthoSlicer3D
from tqdm import tqdm
import skimage
import nibabel
import nibabel.processing
from shutil import copy,move
# from nilearn.image import new_img_like,resample_to_img
from scipy.io import loadmat
from datetime import datetime
import nibabel as nb
# from deepbrain import Extractor
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats.stats import kendalltau
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from matplotlib import pyplot as plt
# from sklearn import LmerRegressor
import pandas as pd
import os
# from pymer4.utils import get_resource_path
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# import palettable
plt.rc('font',family='Times New Roman')
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib import font_manager
from statsmodels.stats.multitest import multipletests
from helper_funcs import *


import zipfile


def normalization(scan):
    # scan = (scan - np.mean(scan)) / np.std(scan)
    scan = (scan - np.min(scan)) / (np.max(scan)-np.min(scan))
    return scan
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, patients,control, transform=None):
        self.patients = patients
        self.control = control
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def normalization(self,scan):
        # scan = (scan - np.mean(scan)) / np.std(scan)
        scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan))
        return scan
    def extend_pad(self,x):
        x=np.expand_dims(x, axis=0)
        x=torch.FloatTensor(x)
        return x
    def __getitem__(self, item):
        # pat=self.patients[item,:,:,:]
        pat=np.load(self.patients[item])
        # contr = self.control[item,:,:,:]
        contr=np.load(self.control[item])
        pat=self.extend_pad(pat)
        contr=self.extend_pad(contr)
        return pat,contr
class MyDataSet_vae(Dataset):
    """自定义数据集"""

    def __init__(self, data_list, transform=None):
        self.data=data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def extend_pad(self,x):
        x=np.expand_dims(x, axis=0)
        x=torch.FloatTensor(x)
        return x
    def __getitem__(self, item):
        # pat=self.patients[item,:,:,:]
        data_npy=np.load(self.data[item])
        data_npy=self.extend_pad(data_npy)
        return data_npy
def generator_loader(args):
    pd_subs=os.path.join(args.data,'pd')
    nc_subs = os.path.join(args.data, 'nc')
    pd_subs_list=[os.path.join(pd_subs,i) for i in os.listdir(pd_subs)]
    nc_subs_list = [os.path.join(nc_subs, i) for i in os.listdir(nc_subs)]
    # len_nc=len(nc_subs_list)
    # pd_subs_list=pd_subs_list[:len_nc]
    my_dataset =MyDataSet(pd_subs_list, nc_subs_list)
    data_loader=torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,pin_memory=True,num_workers=args.workers )
    return pd_subs_list,nc_subs_list,data_loader
def generator_loader_vae(args):
    pd_subs=os.path.join(args.data,'pd')
    nc_subs = os.path.join(args.data, 'nc')
    pd_subs_list=[os.path.join(pd_subs,i) for i in os.listdir(pd_subs)]
    nc_subs_list = [os.path.join(nc_subs, i) for i in os.listdir(nc_subs)]
    # len_nc=len(nc_subs_list)
    # pd_subs_list=pd_subs_list[:len_nc]
    pd_subs_list.extend(nc_subs_list)
    my_dataset =MyDataSet_vae(pd_subs_list)
    data_loader=torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,pin_memory=True,num_workers=args.workers)
    return pd_subs_list,data_loader
# Progress Plotting Functions
def read_resample_normalization(i_path,data_list):
    data_i = ants.image_read(i_path)
    data_i = ants.resample_image(data_i, resample_params=(64, 64, 64), use_voxels=True, interp_type=4)
    sample_data_arr = data_i.numpy()
    sample_data_arr = normalization(sample_data_arr)
    data_list.append(sample_data_arr)
    return data_list,sample_data_arr
def selece_resample_zheda_t1_MNI():
    dti_path = '/home/istbi/HD1/chengyu2_dataset/zheda_t1/bet_MNI'
    save = '/home/istbi/HD1/chengyu2_dataset/zheda_t1/bet_MNI_BL_npy'

    data_npy = []
    nc_pd_BL_list = []
    nc_pd_list = []
    save_nc = os.path.join(save, "NC")
    if os.path.exists(save_nc) is False:
        os.mkdir(save_nc)
    save_pd = os.path.join(save, "PD")
    if os.path.exists(save_pd) is False:
        os.mkdir(save_pd)
    for i in os.listdir(dti_path):
        i_name = i.split('.nii.gz')[0]
        nc_pd_list.append(i_name)
        if 'm' not in i.split('mni.nii.gz')[0] and '_2' not in i.split('mni.nii.gz')[0]:
            print(i_name)
            ori_path_i = os.path.join(dti_path, i)
            nc_pd_BL_list.append(i_name)
            data_npy, sample_data_arr = read_resample_normalization(ori_path_i, data_npy)
            if 'PD' in i:
                np.save(os.path.join(save_pd, i_name + ".npy"), sample_data_arr)
            if 'NC' in i:
                np.save(os.path.join(save_nc, i_name + ".npy"), sample_data_arr)
    df_all = pd.DataFrame(np.array(nc_pd_list), columns={'file_name'})
    df_all.to_csv('/home/istbi/HD1/chengyu2_dataset/zheda_t1/zheda_t1_bet_MNI_pd_nc_all.csv', index=False)
    df = pd.DataFrame(np.array(nc_pd_BL_list), columns={'file_name'})
    print(df.shape)
    df.to_csv('/home/istbi/HD1/chengyu2_dataset/zheda_t1/zheda_t1_bet_MNI_pd_nc_BL.csv', index=False)
    np.save('../Data/zheda_t1_bet_MNI_pd_nc_nor.npy', np.array(data_npy))
def select_fast_nii_by_cat12_dbm_iqr():
    # iqr_id = pd.read_csv('/home/istbi/HD1/chengyu2_dataset/PPMI/cvae/cat12_dbm/ppmi_IQR_BL_name_id_variable.csv')
    iqr_id = pd.read_csv('/public/home/zhenglp/dataset/ppmi/ppmi_IQR_BL_name_id_variable.csv')
    # mni_nii = "/home/istbi/HD1/chengyu2_dataset/PPMI/cvae/fast_ori/fast_nii"
    mni_nii='/public/home/zhenglp/DataSet/ppmi_cvae/fast_ori'
    # path_npy = '/home/istbi/HD1/chengyu2_dataset/PPMI/cvae/fast_ori/fast_npy'

    data_list = []
    pd_data = iqr_id[iqr_id['APPRDX'] == 1]
    nc_data = iqr_id[iqr_id['APPRDX'] == 2]
    pd_id = pd_data['file_name'].unique().tolist()
    nc_id = nc_data['file_name'].unique().tolist()
    print(len(pd_id), len(nc_id))
    print(iqr_id.columns.tolist())
    for i in iqr_id['file_name'].values.tolist():
        print(i)
        i_path = os.path.join(mni_nii, i + '_brain.nii.gz')
        data_i = ants.image_read(i_path)
        sample_data = ants.resample_image(data_i, resample_params=(64, 64, 64), use_voxels=True, interp_type=4)
        # sample_data_arr = sample_data.numpy()
        # sample_data_arr = normalization(sample_data_arr)
        sample_data_arr = ants.iMath_normalize(sample_data)
        sample_data_arr = sample_data_arr.numpy()
        print(sample_data_arr.max(), sample_data_arr.min())
        data_list.append(sample_data_arr)
    print(np.array(data_list).shape)
    # np.save('/home/istbi/HD1/chengyu2_dataset/PPMI/cvae/fast_ori/fast_ori_irq.npy', np.array(data_list))
    np.save("/public/home/zhenglp/DataSet/ppmi_cvae/fast_ori_irq.npy", np.array(data_list))
def prepare_suda():
    fs_path='/home/istbi/HD1/chengyu2_dataset/suda_mri/suda_fs_nii_noregisted'
    mni_path = '/home/istbi/HD1/chengyu2_dataset/suda_mri/suda_ants_nii'
    if os.path.exists(mni_path) is False:
        os.mkdir(mni_path)
    # id="/home/istbi/HD1/chengyu2_dataset/suda_mri/suda.xlsx"
    # id_data=pd.read_excel(id)
    npy_path='/home/istbi/HD1/chengyu2_dataset/suda_mri/suda_ants_npy'
    if os.path.exists(npy_path) is False:
        os.mkdir(npy_path)
    data_npy = []
    pd_list = []
    id_list=[]
    template = ants.image_read('../Data/MNI152_T1_2mm_brain.nii.gz')
    for i in os.listdir(fs_path):
        ori_t1 = ants.image_read(os.path.join(fs_path, i))
        registed_t1 = ants.registration(fixed=template, moving=ori_t1, type_of_transform='Rigid')['warpedmovout']
        registed_t1.to_filename(os.path.join(mni_path, i.split("_")[0] + "_MNI.nii.gz"))
        sample_data = ants.resample_image(registed_t1, resample_params=(64, 64, 64), use_voxels=True, interp_type=4)
        sample_data_arr = ants.iMath_normalize(sample_data)
        sample_data_arr = sample_data_arr.numpy()
        np.save(os.path.join(npy_path, i.split("_")[0] + ".npy"), sample_data_arr)
        data_npy.append(sample_data_arr)
    print(np.array(data_npy).shape)#(71, 64, 64, 64)
    id_suda=pd.read_csv("/home/istbi/HD1/chengyu2_dataset/suda_mri/suda_id.csv",dtype=object)
    id_suda['file_name'] = id_suda['file_name'].apply(str)
    print(id_suda['file_name'].head())
    for i in id_suda['file_name'].values:
        data_npy.append(np.load(os.path.join(npy_path,i+'.npy')))
    print(np.array(data_npy).shape)
    np.save(os.path.join('/home/istbi/HD1/chengyu2_dataset/suda_mri', 'suda_pd_ants.npy'), np.array(data_npy))


def prepare_registed_ants_huashan():
    no_registed='../huashan_fs_nii_no_registed'
    MNI_path='../huashan_fs_nii_MNI_ant'
    npy_path='../huashan_fs_nii_MNI_ant_npy'
    id_list="../huashan_nc_pd_score.csv"
    id_list=pd.read_csv(id_list)
    if os.path.exists(MNI_path) is False:
        os.mkdir(MNI_path)
    if os.path.exists(npy_path) is False:
        os.mkdir(npy_path)
    template = ants.image_read('../Data/MNI152_T1_2mm_brain.nii.gz')
    data_npy=[]
    for i in os.listdir(no_registed):
        id=i.split('.nii.gz')[0]
        ori_t1 = ants.image_read(os.path.join(no_registed, i))
        registed_t1 = ants.registration(fixed=template, moving=ori_t1, type_of_transform='Rigid')['warpedmovout']
        registed_t1.to_filename(os.path.join(MNI_path, id + "_MNI.nii.gz"))
        sample_data = ants.resample_image(registed_t1, resample_params=(64, 64, 64), use_voxels=True, interp_type=4)
        sample_data_arr = ants.iMath_normalize(sample_data)
        sample_data_arr = sample_data_arr.numpy()
        np.save(os.path.join(npy_path, id + ".npy"), sample_data_arr)

    for i in id_list['name'].values:
        data_npy.append(np.load(os.path.join(npy_path, i)))
    print(np.array(data_npy).shape)
    # pd_nc_id = pd.DataFrame(np.array(id_list), columns=["name"])
    # pd_nc_id['file_name'] = pd.DataFrame(np.array([i.split('.npy')[0] for i in pd_nc_id['name'].values]))
    np.save(os.path.join('/home/istbi/HD1/chengyu2_dataset/zuo_pet', 'huashan_pd_nc_78_ant.npy'), np.array(data_npy))


def draw_heatmap_RSA():
    # data_ori=pd.read_csv("/home/daiyx/zhenglp/dataset/PPMI/PPMI/csf_proteins_v2/csf_proteins_fdr_bh_varibales_10_0.05.csv")
    data_ori=pd.read_csv("/home/daiyx/zhenglp/dataset/PPMI/PPMI/csf_proteins_v2/csf_proteins_RSA_varibales_10_0.05_5variables_ols.csv")
    variables_list = ['mean_putamen', 'Serum_NfL', 'updrs3_score', 'moca', 'asyn', 'tau', 'ptau',
                      'abeta','MoCA_slop',"MDS-UPDRS III_slop"]
    draw_variables_list=['key',"Putamen SBR",'Serum_NfL', "MDS-UPDRS III","MoCA","α-Synuclein","T-Tau","P-Tau","A{}42".format(r'$\beta$'),'MoCA slope',"MDS-UPDRS III slope"]
    new_p=["key"]
    new_res=["key"]
    for i in variables_list:
        # new.append(i+"_T")
        new_res.append(i+"_res")
        new_p.append(i + "_P")


    new_res=data_ori[new_res]
    new_res.columns=draw_variables_list
    print(new_res.head())
    key_list=new_res['key'].values.tolist()
    new_res.set_index(new_res['key'],inplace=True)
    print(new_res.head())
    new_res = new_res.drop('key', axis=1)
    # filtered_new_res= new_res[np.abs(new_res) < 0.5]
    # filtered_new_p=new_p[np.abs(new_res) < 0.5]
    select_proteins=[]
    select_proteins2 = []
    new_p = data_ori[new_p]
    # new_p = new_p[new_p['key'].isin(select_proteins)]
    new_p = new_p.set_index('key')

    # print(new_p.values.max(),new_p.values.min()) #0.0571477128828497 -0.0409038425379662
    for i in key_list:#对于每一个蛋白
        eject, pv_fdr = multipletests(new_p.loc[i], alpha=0.05, method="fdr_bh")[:2]
        even_numbers = list(filter(lambda x: x < 0.05, pv_fdr))
        if len(even_numbers) == 10:
            # select_proteins.append(i+'_fdr_bh')
            select_proteins.append(i)
    print(select_proteins)
    print(len(select_proteins))
    # print(select_proteins2)
    # print(len(select_proteins2))

    new_res=new_res.loc[select_proteins]
    print(new_res.shape)
    print(new_res.head())
    new_res = new_res.rename_axis('', axis=0)
    print(new_res.shape)
    print(new_res.head())

    new_p=new_p.loc[select_proteins]
    new_p_r=new_p.transpose()

    #
    new_res_trans=new_res.transpose()
    #
    plt.figure(figsize=(10, 8), dpi=150)
    plt.rcParams['font.family'] = 'Times New Roman'
    # sns.set(font_scale=1.2)
    cmap = sns.color_palette("coolwarm")#"coolwarm" "RdBu"
    max_r=round(new_res_trans.values.max(),2)
    min_r=round(new_res_trans.values.min(),2)
    print(max_r,min_r)
    cluster_map=sns.clustermap(data=new_res_trans,  #method='single',
                   metric='euclidean',
                   # cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
                   linewidths=.75,
                   figsize=(10, 8),
                   vmax=0.05,
                   vmin=-0.05,
                   cbar_kws={'aspect': 5,'shrink': 0.5},#
                  # cbar_kws={"vmin": 0, "vmax": 1}
                   cmap=cmap,
                   row_cluster=False,  #
                   col_cluster=True,  #

                   # cbar_pos=(0, .2, .03, .4)#
                   )
    # print("check")
    ax = cluster_map.ax_heatmap
    # # reordered_rows = new_r_trans.index[cluster_map.dendrogram_row.reordered_ind]
    reordered_columns = new_res_trans.columns[cluster_map.dendrogram_col.reordered_ind]
    # print(reordered_columns)
    # # print(reordered_rows)
    # new_p_r=new_p.reindex(reordered_rows)
    new_p_r=new_p_r.reindex(columns=reordered_columns)

    thresholds = [0.001, 0.01, 0.05]  #
    for i in range(new_res_trans.shape[0]):
        for j in range(new_res_trans.shape[1]):
            if new_p_r.iloc[i, j]<0.001:
                ax.text(j + 0.5, i + 0.5, "***", fontsize=12, ha='center', va='center', color='black')
            if new_p_r.iloc[i, j]>0.001 and new_p_r.iloc[i, j]<0.01:
                ax.text(j + 0.5, i + 0.5, "**", fontsize=12, ha='center', va='center', color='black')
            if new_p_r.iloc[i, j]>0.01 and new_p_r.iloc[i, j]<0.05:
                ax.text(j + 0.5, i + 0.5, "*", fontsize=12, ha='center', va='center', color='black')
    plt.savefig("../cvae_proteins_variables_RSA_fdr_heatmap_11.29_5varibales_ols.jpg",dpi=300)
    plt.show()

if __name__ == '__main__':
        ori="../PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv"
        serum='../serum_nfl_BL.csv'



        selece_resample_zheda_t1_MNI()

        save_path='/mnt/raid/chengyu2_dataset/PPMI/cvae/ppmi_t1_MNI'

        select_fast_nii_by_cat12_dbm_iqr()

        prepare_suda()
        prepare_registed_ants_huashan()


        draw_heatmap_RSA()
        # draw_scatter()
        # selecet_moca_u3_by_commmond_id()
        # linear_fit()
        print("done")


