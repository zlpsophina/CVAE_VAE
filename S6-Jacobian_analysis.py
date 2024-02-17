import numpy as np
import pandas as pd
import ants
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as gmm
# from umap import UMAP
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.stats import kendalltau,pearsonr
import nibabel as nib
# from enigmatoolbox.utils.parcellation import parcel_to_surface
# from enigmatoolbox.plotting import plot_cortical
# from enigmatoolbox.plotting import plot_subcortical
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

# from statannot import add_stat_annotation
import seaborn as sns
from sklearn.cluster import KMeans

def load_data():

    df=pd.read_csv("../ppmi_IQR_BL_name_id_variable.csv")
    patients = df['APPRDX'].values == 1#1#"PD"
    df_asd = df.iloc[patients]
    # data = np.load('../Data/ppmi_fast_freesurfer_pd_nc.npy')
    return df,patients,df_asd

def load_pca(df_asd,est_pca,normed_Js,save_path):
    subs = df_asd['file_name'].values
    fn_temp=os.path.join(normed_Js,'jacobians/normed_Js_masked/{}_normed_Js_masked.nii')
    ims = [ants.image_read(fn_temp.format(sub)) for sub in tqdm(subs)]
    flatmap = np.array([im.numpy().flatten() for im in ims])
    evox = ((flatmap ** 2).sum(axis=0) != 0)
    flatmap = flatmap[:, evox]  # only analyze voxels with values > 0
    flatmap = flatmap - flatmap.mean(axis=0)  # center each voxel at zero
    print("flatmap.shape",flatmap.shape) #flatmap.shape (330, 262144)
    mean_J = ims[0].new_image_like(np.array([im.numpy() for im in ims]).mean(axis=0))

    if est_pca:
        ns = flatmap.shape[0]
        j_pca_loso = np.array(
            [PCA().fit(flatmap[np.arange(ns) != s]).transform(flatmap[s, :][np.newaxis, :]) for s in tqdm(range(ns))])
        # np.save(file='../Data/j_pca_loso.npy', arr=j_pca_loso)
        np.save(os.path.join(save_path,"j_pca_loso.npy"),arr=j_pca_loso)
    else:
        # j_pca_loso = np.load('../Data/j_pca_loso.npy')
        j_pca_loso = np.load(os.path.join(save_path,"j_pca_loso.npy"))
    # which PCA to use
    j_pca_loso_2 = j_pca_loso[:, 0, 0:2]
    j_pca = j_pca_loso_2
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(j_pca[:, 0], j_pca[:, 1], alpha=.3, s=100, color=np.array([123, 106, 149]) / 255)
    plt.xticks([])
    plt.yticks([])


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
        ax.spines[axis].set_color("white")
        # ax.spines[axis].set_zorder(0)
    plt.savefig(os.path.join(save_path,'PCA_scatterplot.pdf'))
    plt.close()

    return ims,j_pca

def recon_jacobian(ims,j_pca):
    corr = kendalltau
    jacobians_flat = np.array([im.numpy().flatten() for im in ims])
    print("jacobians_flat.shape",jacobians_flat.shape) #jacobians_flat.shape (330, 262144)
    # DO X AXIS
    res = np.array([corr(jacobians_flat[:, v], j_pca[:, 0]) for v in tqdm(range(jacobians_flat.shape[1]))])
    xcorr_r = res[:, 0]
    xcorr_p = res[:, 1]
    print("xcorr_p",xcorr_p.shape) #ycorr_p (262144,)

    # DO Y AXIS
    res = np.array([corr(jacobians_flat[:, v], j_pca[:, 1]) for v in tqdm(range(jacobians_flat.shape[1]))])
    ycorr_r = res[:, 0]
    ycorr_p = res[:, 1]
    print("ycorr_p", ycorr_p.shape)
    xcorr_r_nii = ims[0].new_image_like(xcorr_r.reshape((64, 64, 64)))

    # BONF CORRECT MAPS
    xcorr_r_threshed = xcorr_r.copy()
    ycorr_r_threshed = ycorr_r.copy()

    p_thresh = .05 / (~np.isnan(xcorr_p)).sum()
    print(p_thresh) #1.9073486328125e-07

    xcorr_r_threshed[xcorr_p > p_thresh] = np.nan
    ycorr_r_threshed[ycorr_p > p_thresh] = np.nan

    print((~np.isnan(xcorr_p)).sum()) #262144
    print((abs(xcorr_r_threshed[xcorr_p < p_thresh])).min())  # minimum x value to be significant
    print((abs(ycorr_r_threshed[ycorr_p < p_thresh])).min())  # minimum y value to be significant

    # Show tau value ranges (neg and pos)
    print((xcorr_r_threshed[xcorr_r_threshed < 0].min().round(2), xcorr_r_threshed[xcorr_r_threshed < 0].max().round(2)))
    print((xcorr_r_threshed[xcorr_r_threshed > 0].min().round(2), xcorr_r_threshed[xcorr_r_threshed > 0].max().round(2)))

    print((ycorr_r_threshed[ycorr_r_threshed < 0].min().round(2), ycorr_r_threshed[ycorr_r_threshed < 0].max().round(2)))
    print((ycorr_r_threshed[ycorr_r_threshed > 0].min().round(2), ycorr_r_threshed[ycorr_r_threshed > 0].max().round(2)))

    temp = ants.image_read('../Data/MNI152_T1_2mm_brain.nii.gz')
    temp = temp.resample_image(resample_params=(64, 64, 64), use_voxels=True, interp_type=1)

    xcorr_nii = ims[0].new_image_like(
        xcorr_r_threshed.reshape((64, 64, 64)))  # .plot_ortho(flat=True,black_bg=True,cmap='bwr')
    ycorr_nii = ims[0].new_image_like(
        ycorr_r_threshed.reshape((64, 64, 64)))  # .plot_ortho(flat=True,black_bg=True,cmap='bwr')
    temp.plot_ortho(xcorr_nii, flat=True, black_bg=False, cmap='gray', overlay_cmap='bwr',title='xcorr_nii')
    temp.plot_ortho(ycorr_nii, flat=True, black_bg=False, cmap='gray', overlay_cmap='bwr',title='ycorr_nii')#bg_thresh_quant=0.01,

    ycorr_r_threshed[np.isnan(ycorr_r_threshed)] = 0
    xcorr_r_threshed[np.isnan(xcorr_r_threshed)] = 0
    print("ycorr_r_threshed",ycorr_r_threshed.shape)
    print("xcorr_r_threshed", xcorr_r_threshed.shape)

    print(np.nanmin(xcorr_r).round(3), np.nanmean(xcorr_r).round(3), np.nanmax(xcorr_r).round(3))
    print(np.nanmin(ycorr_r).round(3), np.nanmean(ycorr_r).round(3), np.nanmax(ycorr_r).round(3))

    plt.hist(jacobians_flat[:, ~np.isnan(xcorr_p)].mean(axis=1))
    plt.savefig(os.path.join(save_path,'hist_Jacobians_flat.pdf'))
    return xcorr_r,ycorr_r,xcorr_r_threshed,ycorr_r_threshed,jacobians_flat

    # %%
def CORR_PCA_SYMOTOMS(df,patients,j_pca,keys,save_path):
    corr = kendalltau
    # keys = ['ADOS_Comm', 'ADOS_Social', 'ADOS_StBeh', 'AgeAtScan', 'Sex', 'FIQ']
    # keys=keys_list
    npcs = 2  # How many PCs
    # Make dataframe
    col1 = list()
    col2 = list()
    for key in keys:
        for i in ['r', 'p', 'df']:
            col1.append(key)
            col2.append(i)
    # columns = [[key,key],['r','p']]
    columns = [col1, col2]
    res_corr = pd.DataFrame(np.zeros((npcs, len(col1))), columns=columns)

    for key in keys:
        print('key',key)
        for pc in range(npcs):
            vec_behav = df[key].values[patients]
            # vec_pca = j_pca_loso[:,0,pc]
            vec_pca = j_pca[:, pc]
            e = np.isnan(vec_behav)
            r, p = corr(vec_behav[~e], vec_pca[~e])
            res_corr.loc[pc].at[(key, 'r')] = r
            res_corr.loc[pc].at[(key, 'p')] = p
            res_corr.loc[pc].at[(key, 'df')] = len(vec_behav[~e]) - 2

    pd.options.display.max_columns = None
    print("res_corr")
    print(res_corr)
    res_corr.to_csv(os.path.join(save_path,"res_corr.csv"),index=False)

    # these_keys = ['ADOS_Comm', 'ADOS_Social', 'ADOS_StBeh', 'AgeAtScan', 'Sex', 'FIQ']
    these_keys=keys
    for pc in range(res_corr.shape[0]):
        for key in these_keys:
            p = res_corr[key]['p'][pc]
            r = res_corr[key]['r'][pc]
            dgf = res_corr[key]['df'][pc]

            if p < .05:
                print(f'PC{pc} | {key} | $\\tau$({int(dgf)}) = {r:.2f}, p = {p:.3f}')

def split_into_positive_negative(ims,xcorr_r,ycorr_r,xcorr_r_threshed,ycorr_r_threshed):

    # c1 = ants.image_read('../Data/c1Atlas_brain_2mm.nii').resample_image(resample_params=(64, 64, 64), use_voxels=True,
    #                                                                      interp_type=1)
    # c2 = ants.image_read('../Data/c2Atlas_brain_2mm.nii').resample_image(resample_params=(64, 64, 64), use_voxels=True,
    #                                                                      interp_type=1)

    temp = ants.image_read('../Data/MNI152_T1_2mm_brain.nii.gz')
    temp = temp.resample_image(resample_params=(64, 64, 64), use_voxels=True, interp_type=1)
    xcorr_r_nii = ims[0].new_image_like(xcorr_r.reshape((64, 64, 64)))#.to_filename(os.path.join(save_path,'xcorr.nii'))
    ycorr_r_nii = ims[0].new_image_like(ycorr_r.reshape((64, 64, 64)))#.to_filename(os.path.join(save_path,'ycorr.nii'))
    xcorr_r_nii.to_filename(os.path.join(save_path,'xcorr.nii'))
    ycorr_r_nii.to_filename(os.path.join(save_path,'ycorr.nii'))
    xcorr_r_threshed_nii = ims[0].new_image_like(
        xcorr_r_threshed.reshape((64, 64, 64)))
    xcorr_r_threshed_nii.to_filename(os.path.join(save_path,'xcorr-bonf.nii'))
    ycorr_r_threshed_nii = ims[0].new_image_like(
        ycorr_r_threshed.reshape((64, 64, 64)))
    ycorr_r_threshed_nii.to_filename(os.path.join(save_path,'ycorr-bonf.nii'))
    xcorr_r_nii_nii_pos = xcorr_r_nii.copy()
    xcorr_r_nii_nii_neg = xcorr_r_nii.copy()
    xcorr_r_nii_nii_pos[xcorr_r_nii < 0] = 0
    xcorr_r_nii_nii_neg[xcorr_r_nii > 0] = 0

    ycorr_r_nii_nii_pos = ycorr_r_nii.copy()
    ycorr_r_nii_nii_neg = ycorr_r_nii.copy()
    ycorr_r_nii_nii_pos[ycorr_r_nii < 0] = 0
    ycorr_r_nii_nii_neg[ycorr_r_nii > 0] = 0

    xcorr_r_nii_nii_neg = xcorr_r_nii_nii_neg.new_image_like(abs(xcorr_r_nii_nii_neg.numpy()))
    ycorr_r_nii_nii_neg = ycorr_r_nii_nii_neg.new_image_like(abs(ycorr_r_nii_nii_neg.numpy()))

    xcorr_r_nii_nii_pos[np.isnan(xcorr_r_nii_nii_pos.numpy())] = 0
    xcorr_r_nii_nii_neg[np.isnan(xcorr_r_nii_nii_neg.numpy())] = 0
    ycorr_r_nii_nii_pos[np.isnan(ycorr_r_nii_nii_pos.numpy())] = 0
    ycorr_r_nii_nii_neg[np.isnan(ycorr_r_nii_nii_neg.numpy())] = 0


    temp.plot_ortho(xcorr_r_nii_nii_pos,flat=True,black_bg=True,title='xcorr_r_nii_nii_pos',
                    filename=os.path.join(save_path,'xcorr_r_nii_nii_pos.jpg'))


    temp.plot_ortho(xcorr_r_nii_nii_neg, flat=True,black_bg=True,  title='xcorr_r_nii_nii_neg',
                    filename=os.path.join(save_path, 'xcorr_r_nii_nii_neg.jpg'))


    temp.plot_ortho(ycorr_r_nii_nii_pos, flat=True,black_bg=True,  title='ycorr_r_nii_nii_pos',
                    filename=os.path.join(save_path, 'ycorr_r_nii_nii_pos.jpg'))

    temp.plot_ortho(ycorr_r_nii_nii_neg, flat=True,black_bg=True,  title='ycorr_r_nii_nii_neg',
                    filename=os.path.join(save_path, 'ycorr_r_nii_nii_neg.jpg'))

    #画出卡阈值之后的pos neg
    xcorr_r_threshed_nii_pos = xcorr_r_threshed_nii.copy()
    xcorr_r_threshed_nii_neg = xcorr_r_threshed_nii.copy()
    xcorr_r_threshed_nii_pos[xcorr_r_threshed_nii_pos < 0] = 0
    xcorr_r_threshed_nii_neg[xcorr_r_threshed_nii_neg > 0] = 0

    ycorr_r_threshed_nii_pos = ycorr_r_threshed_nii.copy()
    ycorr_r_threshed_nii_neg = ycorr_r_threshed_nii.copy()
    ycorr_r_threshed_nii_pos[ycorr_r_threshed_nii_pos < 0] = 0
    ycorr_r_threshed_nii_neg[ycorr_r_threshed_nii_neg > 0] = 0

    ycorr_r_threshed_nii_neg = ycorr_r_threshed_nii_neg.new_image_like(abs(ycorr_r_threshed_nii_neg.numpy()))
    xcorr_r_threshed_nii_neg = xcorr_r_threshed_nii_neg.new_image_like(abs(xcorr_r_threshed_nii_neg.numpy()))

    temp.plot_ortho(xcorr_r_threshed_nii_pos,flat=True,black_bg=True,  title='xcorr_r_threshed_nii_pos',
                    filename=os.path.join(save_path, 'xcorr_r_threshed_nii_pos.jpg'))
    # xcorr_r_threshed_nii_pos.plot_ortho(flat=True, cmap='hot')
    # plt.savefig(os.path.join(save_path, 'xcorr_r_threshed_nii_pos.jpg'))
    temp.plot_ortho(xcorr_r_threshed_nii_neg,flat=True,black_bg=True,  title='xcorr_r_threshed_nii_neg',
                    filename=os.path.join(save_path, 'xcorr_r_threshed_nii_neg.jpg'))
    # xcorr_r_threshed_nii_neg.plot_ortho(flat=True, cmap='hot')
    # plt.savefig(os.path.join(save_path, 'xcorr_r_threshed_nii_neg.jpg'))

    temp.plot_ortho(ycorr_r_threshed_nii_pos, flat=True,black_bg=True,  title='ycorr_r_threshed_nii_pos',
                    filename=os.path.join(save_path, 'ycorr_r_threshed_nii_pos.jpg'))
    # ycorr_r_threshed_nii_pos.plot_ortho(flat=True, cmap='hot')
    # plt.savefig(os.path.join(save_path, 'ycorr_r_threshed_nii_pos.jpg'))
    temp.plot_ortho(ycorr_r_threshed_nii_neg, flat=True,black_bg=True,  title='ycorr_r_threshed_nii_neg',
                    filename=os.path.join(save_path, 'ycorr_r_threshed_nii_neg.jpg'))
    # ycorr_r_threshed_nii_neg.plot_ortho(flat=True, cmap='hot')
    # plt.savefig(os.path.join(save_path, 'ycorr_r_threshed_nii_neg.jpg'))

    return xcorr_r_threshed_nii_pos,xcorr_r_threshed_nii_neg,ycorr_r_threshed_nii_pos,ycorr_r_threshed_nii_neg

def Volume_increases_decreases_along_PCA_axes(j_pca,jacobians_flat,xcorr_r_threshed_nii_pos,xcorr_r_threshed_nii_neg,ycorr_r_threshed_nii_pos,ycorr_r_threshed_nii_neg):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    v1 = j_pca[:, 0]
    v2 = jacobians_flat[:, xcorr_r_threshed_nii_pos.numpy().flatten() > 0.01].mean(axis=1)
    plt.scatter(v1, v2)
    plt.xlabel('PC 0')
    plt.ylabel('mean J val at x pos')
    plt.title('x pos')
    plt.subplot(2, 2, 2)
    v1 = j_pca[:, 0]
    v2 = jacobians_flat[:, xcorr_r_threshed_nii_neg.numpy().flatten() > 0.01].mean(axis=1)
    plt.scatter(v1, v2)
    plt.xlabel('PC 0')
    plt.ylabel('mean J val at x neg')
    plt.title('x neg')

    plt.subplot(2, 2, 3)
    v1 = j_pca[:, 1]
    v2 = jacobians_flat[:, ycorr_r_threshed_nii_pos.numpy().flatten() > 0.01].mean(axis=1)
    plt.scatter(v1, v2)
    plt.xlabel('PC 1')
    plt.ylabel('mean J val at y pos')
    plt.title('y pos')

    plt.subplot(2, 2, 4)
    v1 = j_pca[:, 1]
    v2 = jacobians_flat[:, ycorr_r_threshed_nii_neg.numpy().flatten() > 0.01].mean(axis=1)
    plt.scatter(v1, v2)
    plt.xlabel('PC 1')
    plt.ylabel('mean J val at y neg')
    plt.title('y neg')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'Volume_increases_decreases_along_PCA_axes.jpg'))
    plt.close()

    # plt.hist(jacobians_flat[:, ~np.isnan(xcorr_p)].mean(axis=1))
def load_normed_JS(normel_Js,df_asd,save_path):
    subs = df_asd['file_name'].values

    fn_temp=os.path.join(normel_Js,'jacobians/normed_Js_masked/{}_normed_Js_masked.nii')
    ims = [ants.image_read(fn_temp.format(sub)) for sub in tqdm(subs)]
    flatmap = np.array([im.numpy() for im in ims])
    print(flatmap.shape)
    mean=flatmap.mean(axis=0)#.astype('float32')
    mean_img_1=ims[0].new_image_like(mean)
    # mean_img_2=ants.from_numpy(mean,spacing=img.spacing,origin=img.origin, direction=img.direction)
    ants.image_write(mean_img_1, os.path.join(save_path,'mean_pd_1.nii.gz'))
    # ants.image_write(mean_img_2,"./step6_Jacobian_analysis_result/mean_pd_2.nii.gz")

def make_std_JS(normel_Js,df_asd,save_path,name):

    subs = df_asd['file_name'].values
    fn_temp = os.path.join(normel_Js, 'jacobians/normed_Js_masked/{}_normed_Js_masked.nii')

    ims = [ants.image_read(fn_temp.format(sub)) for sub in tqdm(subs)]
    flatmap = np.array([im.numpy() for im in ims])
    print(flatmap.shape)
    mean = flatmap.std(axis=0)  # .astype('float32')
    mean_img_1 = ims[0].new_image_like(mean)
    # mean_img_2=ants.from_numpy(mean,spacing=img.spacing,origin=img.origin, direction=img.direction)
    ants.image_write(mean_img_1, os.path.join(save_path, '{}_std.nii.gz'.format(name)))

def mask_JS(normel_Js):
    save_path=os.path.join(normel_Js,'jacobians')
    mask_path=os.path.join(save_path,'normed_Js_masked')
    if os.path.exists(mask_path) is False:
        os.mkdir(mask_path)
    mask=ants.image_read('../Data/MNI152_T1_2mm_brain_mask.nii.gz')#MNI152_T1_1mm_brain_mask.nii.gzReslice_64_MNI152_T1_1mm_Brain_Mask.nii
    mask=ants.resample_image(mask,resample_params=(64, 64, 64), use_voxels=True, interp_type=1)
    # mask = ants.get_mask(mask_img)
    fn_temp=os.path.join(normel_Js,'jacobians/normed_Js')
    for i in os.listdir(fn_temp):
        i_name=i.split('/')[-1].split('.nii')[0]
        i_path=os.path.join(fn_temp,i)
        i_img=ants.image_read(i_path)
        # normed_J = ants.registration(fixed=template, moving=i_img, type_of_transform='Rigid')['warpedmovout']
        masked_i = ants.mask_image(i_img, mask)
        save_name_i=i_name+'_masked.nii'
        ants.image_write(masked_i,os.path.join(mask_path,save_name_i))
def corr_JS_score(save_path,df_pd,keys):
    j_pca_loso = np.load(os.path.join(save_path, "j_pca_loso.npy"))
    print("j_pca_loso",j_pca_loso.shape)
    j_pca_loso_2 = j_pca_loso[:, 0, 0:2]
    j_pca = j_pca_loso_2
    print("j_pca",j_pca.shape)
    kmeans = KMeans(2, random_state=2022)
    kmeans.fit(j_pca)  # 训练模型
    labels = kmeans.labels_  # 预测分类
    print("len(labels)",len(labels))
    patients=df_pd['PATNO'].values.tolist()
    print("patients len",len(patients))
    cat=['PATNO']
    cat.extend(keys)
    # df_asd=
    df_pd=df_pd[cat]
    # print(df_pd.head())
    patients_df = pd.DataFrame(np.array(patients), columns={'PATNO'})
    labels_df = pd.DataFrame(np.array(labels), columns={'cluster'})
    df = pd.concat([patients_df, labels_df], axis=1)
    df =pd.merge(df,df_pd,on='PATNO',how='inner')
    print(df.shape)
    df.to_csv(os.path.join(save_path, 'JS_PATNO_cluster.csv'), index=False)
    plt.scatter(j_pca[:, 0], j_pca[:, 1], c=labels, s=40, cmap='viridis')
    # plt.title(title)
    plt.savefig(os.path.join(save_path, 'JS_kmeans_cluster.jpg'))
    plt.close()
    return df



def draw_boxplot_select_feature(df,p_value_data,n_cluster,save_path):
    # p_value_data=pd.read_csv(T_P_value)
    # print(df.head())
    p_value_data=p_value_data.sort_values(by='P value',ascending=True)
    p_value_data=p_value_data[p_value_data['P value']< 0.05]
    print("p_value_data",p_value_data.shape)
    select_feature_list=p_value_data['feature'].values.tolist()
    p_value_data=p_value_data.set_index('feature')

    count1=df[df['cluster']==1]['cluster'].value_counts()
    count0=df[df['cluster']==0]['cluster'].value_counts()
    print(count0.values[0],count1.values[0])
    df.rename(columns={"cluster":"pattern"},inplace=True)
    # print(df.head())
    # print(len(select_feature_list))
    for i in select_feature_list:
        x = "pattern"
        y = i
        # print(i in df.columns.tolist())
        # print(cat_name)
        order = [(k+1) for k in range(n_cluster)]
        # order=["pattern 1","pattern 2"]
        # print(order)
        leng_list = ["pattern {}".format(i + 1) for i in range(n_cluster)]
        # p_value_list = p_value_data.loc[i]  # 读取该item的所有列值T P值
        # p_value_list = p_value_list.iloc[:, p_value_list.columns.str.contains("P")]
        # p_value = p_value_list.values.flatten().tolist()  # p值转list
        p_value=p_value_data.loc[i,"P value"]
        print("item",i,"p value",p_value)
        text_annot_custom = ["p = {}".format(round(p_value, 6))]
        ax = sns.boxplot(data=df, x=x, y=y)
        ax = sns.swarmplot(x=x, y=y, data=df,color=".25")

        ax.set_xticklabels(["0 (n={})".format(count0.values[0]),"1 (n={})".format(count1.values[0])])

        add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                            box_pairs=[(1, 2)], text_annot_custom=text_annot_custom,
                            perform_stat_test=False, pvalues=[p_value],
                            test=None, text_format='star', loc='inside', verbose=2)
        # add_stat_annotation(ax, data=df, x=x, y=y, order=order,
        #                     box_pairs=[(1, 2), (1, 3), (2, 3)],
        #                     test="t-test_ind", text_format='star', loc='inside', verbose=2)
        # plt.legend(leng_list)
        plt.savefig(os.path.join(save_path, '{}_pattern_pvalues.png'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()
    print("done!!!")




if __name__ == '__main__':
    normel_Js='./step5_make_Jacobian_result_739/cvae_bspline_10_1_0.9_0.001'
    save_path='./step6_Jacobian_analysis_result_739'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    name = normel_Js.split('/')[-1]
    save_path = os.path.join(save_path, name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    keys = ["SITE", "age", "gen", "updrs3_score", "rem", "scopa", "updrs1_score",
            "updrs2_score", "moca", "updrs_totscore", 'TESTVALUE']
    df, patients, df_asd=load_data()
    # #
    mask_JS(normel_Js)
    load_normed_JS(normel_Js,df_asd,save_path)
    ims, j_pca=load_pca(df_asd,est_pca = True,normed_Js=normel_Js,save_path=save_path)
    xcorr_r, ycorr_r, xcorr_r_threshed, ycorr_r_threshed,jacobians_flat=recon_jacobian(ims,j_pca)
    CORR_PCA_SYMOTOMS(df, patients, j_pca,keys, save_path,)
    xcorr_r_threshed_nii_pos,xcorr_r_threshed_nii_neg,ycorr_r_threshed_nii_pos,ycorr_r_threshed_nii_neg=split_into_positive_negative(ims, xcorr_r, ycorr_r, xcorr_r_threshed, ycorr_r_threshed)
    Volume_increases_decreases_along_PCA_axes(j_pca, jacobians_flat, xcorr_r_threshed_nii_pos, xcorr_r_threshed_nii_neg,
                                              ycorr_r_threshed_nii_pos, ycorr_r_threshed_nii_neg)
    normel_Js_volume="../step5_make_Jacobian_result_739/cvae_bspline_10_1_0.9_0.001"
    save_path_volume='../all_dataset_step6_reslut_std'
    ppmi="../step5_make_Jacobian_result_739_suda/cvae_bspline_10_1_0.9_0.001/jacobians"

    make_std_JS(normel_Js,df_asd,save_path_volume,"ppmi_161_nc")
