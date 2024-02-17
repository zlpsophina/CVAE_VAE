import pandas as pd
import numpy as np
# import umap.plot

import ssl
import sys
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as gmm
import scipy
from matplotlib import pyplot as plt
import os
from helper_funcs import *
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster,cut_tree
import matplotlib as mpl
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from sklearn.cluster import AgglomerativeClustering
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib import font_manager
from sklearn import metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
import seaborn as sns
def load_data(npz_path):
    df = pd.read_csv('../ppmi_IQR_BL_name_id_variable.csv')
    patients = df['APPRDX'].values ==1# "PD"
    patients_id=df.iloc[patients]
    patients_id=patients_id['PATNO'].values.tolist()
    print("patients_id",len(patients_id))
    controls = df['APPRDX'].values ==2# "NC"

    data = np.load(npz_path)  # Load latent representations

    return  data,patients,patients_id


def get_feature_one_dataset(data,patients,n_samples):
    get_bic = lambda data_in: np.array([gmm(n_components=i + 1).fit(data_in).bic(data_in) for i in range(n)])
    n = 10  # How many clusters to test
    i = n_samples  # How many samples
    rep = 1
    # BIC for ASD-Specific features
    #data 100 477 16
    # mat=np.zeros((i,n,rep))
    arr_sl = np.zeros((i, n, rep))
    for ii in tqdm(range(i)):
        for jj in range(rep):
            # mat = np.vstack(
            #     (data['salient_vec_sfari'][ii, :, :][cnvs, :], data['salient_vec_abide'][ii, :, :][patients, :]))
            mat= data['salient_vec_abide'][ii, :, :][patients, :]
            # mat=np.nan_to_num(mat.astype(np.float32))
            arr_sl[ii, :, jj] = get_bic(mat)

    # BIC for shared features
    arr_bg = np.zeros((i, n, rep))
    for ii in tqdm(range(i)):
        for jj in range(rep):
            # mat = np.vstack(
            #     (data['background_vec_sfari'][ii, :, :][cnvs, :], data['background_vec_abide'][ii, :, :][patients, :]))
            mat=data['background_vec_abide'][ii, :, :][patients, :]
            arr_bg[ii, :, jj] = get_bic(mat)

    # BIC for VAE features
    arr_vae = np.zeros((i, n, rep))
    for ii in tqdm(range(i)):
        for jj in range(rep):
            # mat = np.vstack((data['vae_vec_sfari'][ii, :, :][cnvs, :], data['vae_vec_abide'][ii, :, :][patients, :]))
            mat=data['vae_vec_abide'][ii, :, :][patients, :]
            arr_vae[ii, :, jj] = get_bic(mat)

    return arr_sl,arr_bg,arr_vae,n

def plot_clustering_result(arr_sl,arr_bg,arr_vae,n,save_path):
    plot_mats = [arr_sl, arr_bg, arr_vae]
    plot_ttls = ['Specific feature', 'Shared feature', 'VAE']
    for i in range(3):
        plot_mat = plot_mats[i]
        plot_ttl = plot_ttls[i]

        xs = np.arange(n)
        figsize = np.array((185, 211)) / 211 * 5
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        m = plot_mat[:, :, 0].mean(axis=0)
        se_min = plot_mat[:, :, 0].min(axis=0)
        se_max = plot_mat[:, :, 0].max(axis=0)
        plt.plot(xs, m, 'k-', linewidth=2.5)
        plt.fill_between(xs, y1=se_min, y2=se_max, alpha=.3, facecolor=[0, 0, 0])

        plt.yticks(fontsize=14, fontweight='bold')
        plt.xticks(xs, labels=xs + 1, fontsize=14, fontweight='bold')
        plt.xlabel('Number of clusters', fontsize=14, fontweight='bold')
        plt.ylabel('BIC\nlower is better', fontsize=14, fontweight='bold')

        line_width = 2.5
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)

        plt.title(plot_ttl, fontsize=14 * 1.5, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,'step4_clustering_BIC_result_{}.jpg'.format(plot_ttl)))
        plt.close()

def plot_umap(embedding_hp1,save_path,cat):
    ig, ax = plt.subplots(figsize=(6, 4))
    contour_c = '#444444'
    plt.xlim([np.min(embedding_hp1[:, 0]) - 0.5, np.max(embedding_hp1[:, 0]) + 1.5])
    plt.ylim([np.min(embedding_hp1[:, 1]) - 0.5, np.max(embedding_hp1[:, 1]) + 0.5])
    labelsize = 2
    plt.xlabel('UMAP 1', fontsize=labelsize)
    plt.ylabel('UMAP 2', fontsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.scatter(embedding_hp1[:, 0], embedding_hp1[:, 1], lw=0, c='#D55E00', label='HPK1', alpha=1.0, s=18, marker="o",
                edgecolors='k', linewidth=2)
    # leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=1.00)
    # leg.get_frame().set_alpha(0.9)
    # for i in range(len(patients_id)):
    #     ax.annotate(i,(embedding_hp1[i, 0], embedding_hp1[i, 1]),size=4)
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(os.path.join(save_path,'umap_cluster_{}.pdf'.format(cat)))
    plt.close()

def umap_reduce(data,patients,patients_id,save_path):
    # print("patients_i_id",len(patients_id))
    salient=data['salient_vec_abide'][:,patients,:]
    background=data['background_vec_abide'][:,patients,:]
    vae_vec=data['vae_vec_abide'][:,patients,:]
    print("salient",salient.shape) #salient (100, 477, 16)
    salient=np.mean(salient,axis=0)
    # print("")
    background=np.mean(background,axis=0)
    vae_vec=np.mean(vae_vec,axis=0)
    # salient=np.array([])
    # salient=np.nan_to_num(salient.astype(np.float32))
    # background = np.nan_to_num(background.astype(np.float32))
    # vae_vec = np.nan_to_num(vae_vec.astype(np.float32))
    # print("salient", salient.shape)

    # cluster_latent(salient_emb,patients_id,'Specific_vec_pd_umap',save_path,'UMAP')
    # cluster_latent(background_emb,patients_id,'share_feature_vec_pd_umap',save_path)
    # cluster_latent(vae_vec_emb,patients_id,'vae_vec_pd_umap',save_path)

    salient_emb_pca=dim_reduce(salient,method='pca',n_neighbor=10,min_dist=0.3)
    # background_emb_pca = dim_reduce(background, method='pca', n_neighbor=10, min_dist=0.3)
    # vae_vec_emb_pca = dim_reduce(vae_vec, method='pca', n_neighbor=10, min_dist=0.3)
    choose_best_cluster_N(salient_emb_pca,'Specific_vec_pd', save_path)

    cluster_latent(salient, patients_id, 'Specific_vec_pd', save_path, 'Specific_vec_pd')

def cluster_latent(data,patients,title,save_path,methods):
    kmeans = KMeans(2, random_state=2022)
    kmeans.fit(data)  # 训练模型
    # labels = kmeans.predict(data)  # 预测分类
    # print(len(labels))
    labels=kmeans.labels_
    centroids = kmeans.cluster_centers_
    patients_df=pd.DataFrame(np.array(patients),columns={'PATNO'})
    labels_df=pd.DataFrame(np.array(labels),columns={'cluster'})
    df=pd.concat([patients_df,labels_df],axis=1)
    # print(df.head())
    df.to_csv(os.path.join(save_path,'{}_PATNO_cluster_kmeans.csv'.format(title)),index=False)
    pal = sns.color_palette()
    plt.figure(figsize=(8, 6))

    pal = sns.color_palette()

    colors = list([pal[1], pal[2]])

    cm = mpl.colors.ListedColormap(colors)  # darkorange['darkorange', 'g']
    for i in range(2):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}',s=40, cmap=cm,alpha=0.8)


    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=80, c='black', label='Centroids')
    plt.title('KMeans Clustering')
    plt.legend()
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.savefig(os.path.join(save_path,'{}_kmeans_cluster_kmeans.jpg'.format(title)),dpi=300)

def choose_best_cluster_N(data,title,save_path):
    # Choose optimal K by elbow method
    sse = []
    for k in range(1, 10):
        model = KMeans(n_clusters=k, random_state=1, n_init=20)
        model.fit(data)
        sse.append(model.inertia_)

    model = KMeans(n_clusters=2, random_state=2022, n_init=20)
    model.fit(data)
    sns.scatterplot(x=data[:, 0], y=data[:, 1], data=data[model.labels_ == 0], color='k', label='y=0')
    sns.scatterplot(x=data[:, 0], y=data[:, 1], data=data[model.labels_ == 1], color='b', label='y=1')
    # sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 2], color='cornflowerblue', label='y=2')
    plt.legend()
    plt.title('Estimated Clusters (K=2)')
    plt.savefig(os.path.join(save_path, '{}_K-means_Clustering_show.jpg'.format(title)))
    plt.close()

    # print(sse)
    plt.plot(range(1, 10), sse, 'o-')
    plt.axhline(sse[1], color='k', linestyle='--', linewidth=1)
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.title('K-means Clustering')
    plt.savefig(os.path.join(save_path, '{}_K-means_Clustering_SSE.jpg'.format(title)),dpi=300)
    plt.close()
    # AIC
    # Choose optimal K by AIC
    aic = sse + 2 * 2 * np.arange(1, 10)
    # min(aic)
    # np.argmin(aic)
    plt.plot(range(1, 10), aic, 'o-')
    plt.axvline(np.argmin(aic) + 1, color='k', linestyle='--', linewidth=1)
    plt.xlabel('K')
    plt.ylabel('AIC')
    plt.title('K-means Clustering AIC')
    plt.savefig(os.path.join(save_path, '{}_K-means_Clustering_AIC.jpg'.format(title)),dpi=300)
    plt.close()
    # BIC
    # Choose optimal K by BIC
    bic = sse + 2 * np.log(100) * np.arange(1, 10)
    # bic
    # min(bic)
    np.argmin(bic)
    plt.plot(range(1, 10), bic, 'o-')
    plt.axvline(np.argmin(bic) + 1, color='k', linestyle='--', linewidth=1)
    plt.xlabel('K')
    plt.ylabel('BIC')
    plt.title('K-means Clustering BIC')
    plt.savefig(os.path.join(save_path, '{}_K-means_Clustering_BIC.jpg'.format(title)),dpi=300)
    plt.close()

def T_test_feature(data,df_nc,feature_type,save_path):
    # feature_list=data.columns.tolist()
    # feature_list.remove('gen')
    # feature_list.remove('race')
    P=[]
    T=[]

    data=data.drop('PATNO',axis=1)
    # print(data['cluster'])
    df1 = data[data['cluster'] == 0]
    df2 = data[data['cluster'] == 1]
    df1 =df1.drop('cluster',axis=1)
    df2 = df2.drop('cluster', axis=1)
    feature_list=df1.columns.tolist()
    data_df = pd.DataFrame(feature_list, columns=['feature'])
    print(len(feature_list))

    for i in feature_list:
        df1_group=df1[i].dropna()
        df2_group=df2[i].dropna()
        if i not in ['gen']:
            t, pvalue=stats.mannwhitneyu(df1_group, df2_group)
            # s, sp = stats.levene(df1_group, df2_group)
            # if sp > 0.05:
            #     t,pvalue = stats.ttest_ind(a=df1_group, b=df2_group, equal_var=True)
            # else:
            #     t,pvalue = stats.ttest_ind(a=df1_group, b=df2_group, equal_var=False)
            # p.append(1-pvalue)

        else:
            c_tab = pd.crosstab(data[i], data['cluster'], margins=True)
            f_obs = np.array([c_tab.iloc[0][0:2].values,
                              c_tab.iloc[1][0:2].values])
            chi2, pval, dof, expected = stats.chi2_contingency(f_obs)
            t, pvalue=chi2,pval
        P.append(pvalue)
        T.append(t)
    print(P[:5])
    reject, pvalscorr = multipletests(P, alpha=0.05, method="fdr_bh")[:2]
    data_df['T value']=pd.DataFrame(T)
    data_df['P value']=pd.DataFrame(P)
    data_df['reject value']=pd.DataFrame(reject)
    data_df['P_corr value']=pd.DataFrame(pvalscorr)
    # data_df = pd.concat([data_df, pd.DataFrame(reject, columns=['reject value'])], axis=1)
    # data_df= pd.concat([data_df, pd.DataFrame(pvalscorr, columns=['P_corr value'])], axis=1)


    for j in range(2):
        A = data[data["cluster"] == j]
        A_mean = []
        A_std = []
        for cat in feature_list:
            A_data = A[cat]
            A_data = A_data.dropna()
            A_mean.append(A_data.mean())
            A_std.append(A_data.std())
        data_df = pd.concat([data_df, pd.DataFrame(A_mean, columns=['{}_mean'.format(j)])], axis=1)
        data_df = pd.concat([data_df, pd.DataFrame(A_std, columns=['{}_std'.format(j)])], axis=1)
    nc_mean = []
    nc_std = []
    for cat in feature_list:
        nc_data = df_nc[cat]
        nc_data = nc_data.dropna()
        nc_mean.append(nc_data.mean())
        nc_std.append(nc_data.std())
    data_df = pd.concat([data_df, pd.DataFrame(nc_mean, columns=['{}_mean'.format('NC')])], axis=1)
    data_df = pd.concat([data_df, pd.DataFrame(nc_std, columns=['{}_std'.format('NC')])], axis=1)

    data_df.to_csv(os.path.join(save_path, "T_P_values_{}.csv".format(feature_type)), index=False)

    return data_df

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
    df.rename(columns={"cluster":"subgroup"},inplace=True)
    # print(df.head())
    # print(len(select_feature_list))
    for i in select_feature_list:
        x = "subgroup"
        y = i
        # print(i in df.columns.tolist())
        # print(cat_name)
        order = [(k+1) for k in range(n_cluster)]
        # order=["pattern 1","pattern 2"]
        # print(order)
        leng_list = ["subgroup {}".format(i + 1) for i in range(n_cluster)]
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
        plt.savefig(os.path.join(save_path, '{}_subgroup_pvalues.png'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()
    print("done!!!")

def get_all_list():
    cat_list = ['updrs_totscore', 'updrs2_score', 'updrs3_score', 'MSEADLG', 'bjlot', 'hvlt_immediaterecall', 'lns',
                'sft', 'SDMTOTAL', 'moca', 'updrs1_score', 'gds', 'stai', 'quip', 'ess', 'rem', 'mean_caudate',
                'mean_putamen', 'mean_striatum']# 'abeta', 'asyn', 'tau', 'ptau'
    score_list = ['age', 'EDUCYRS', 'ageonset', 'td_pigd', 'tremor',
                  'hvlt_discrimination', 'hvlt_retention', ]
    score_list_con = ['scopa_cv', 'scopa_ur', 'scopa_gi', 'scopa_therm', 'scopa_pm', 'scopa_sex', 'scopa',
                      'stai_state', 'quip_gamble', 'quip_sex', 'quip_buy', 'quip_eat', 'quip_hobby', 'quip_pund',
                      'quip_walk', 'quip_any',
                      'CAUDATE_R', 'CAUDATE_L', 'PUTAMEN_R', 'PUTAMEN_L', ]#'tau_ab', 'ptau_ab','ab_asyn', 'tau_asyn', 'ptau_asyn', 'ptau_tau'
    binary_cat = ['gen', 'race']
    csf_list=['abeta','asyn','tau','ptau','tau_ab','ptau_ab','ab_asyn','tau_asyn','ptau_asyn','ptau_tau']
    all_list = []
    all_list.extend(cat_list)
    all_list.extend(score_list)
    all_list.extend(score_list_con)
    all_list.extend(binary_cat)
    return all_list,csf_list

def merge_all_feature(cave_result, aal2_path, V1_path, socre_path, csf_path,cat_list,csf_list):
    df_data = pd.read_csv(cave_result)
    aal_data = pd.read_csv(aal2_path)
    aal_data = aal_data.drop(["time_id", 'participant_id'], axis=1)
    merge_data=pd.merge(df_data,aal_data,on='PATNO',how='left')

    merge_data_aal=merge_data

    V1_data = pd.read_csv(V1_path)
    merge_data = pd.merge(merge_data, V1_data, on='PATNO', how='left')
    merge_data_v1 = pd.merge(df_data,V1_data,on='PATNO',how='left')
    # print("merge v1", merge_data.shape)

    score = pd.read_csv(socre_path)
    stage_score = score[(score["EVENT_ID"] == "BL") & (score['APPRDX'] < 3)]  # 只获取baseline 原先的表 男1 女 2 ，转化为男0 女1
    stage_score["gen"] = stage_score['gen'].map(lambda gen: gen - 1)
    stage_score["APPRDX"] = stage_score['APPRDX'].map(lambda d: -1 if d == 2 else d)  # Pd 1 nc=2 转化为pd=1 nc=-1
    stage_score['race'] = stage_score['race'].map(lambda r: r if r == 1 else 0)
    stage_score.rename(columns={'APPRDX': 'diagnosis'}, inplace=True)
    # stage_score = stage_score[["PATNO", "diagnosis", "SITE", "EDUCYRS", "age", "gen", "race"]]
    cat_list.extend(['PATNO'])
    stage_score_cat = stage_score[cat_list]
    merge_data = pd.merge(merge_data, stage_score_cat, on='PATNO', how='left')
    merge_data_score= pd.merge(df_data,stage_score_cat,on='PATNO',how='left')
    # print("merge score", merge_data.shape)

    csf_list.extend(['PATNO'])
    csf_score=stage_score[csf_list]
    csf_data = pd.read_csv(csf_path)
    merge_data = pd.merge(merge_data, csf_data, on='PATNO', how='left')
    merge_data=pd.merge(merge_data,csf_score,on='PATNO',how='left')
    merge_data_csf = pd.merge(df_data, csf_data, on='PATNO', how='left')
    merge_data_csf = pd.merge(merge_data_csf, csf_score, on='PATNO', how='left')
    # print("merge csf", merge_data.shape)
    return [merge_data,merge_data_aal,merge_data_v1,merge_data_score,merge_data_csf]

def follow_errorbar_score_from_ori(ori_path,cave_result,n_cluster,save_path,cat):
    #这个函数是对总表里面的score 画follow up的数据
    score_data=pd.read_csv(ori_path)
    # print("ori count patno",len(score_data["PATNO"].unique())) #ori count patno 683
    score_data=score_data[['PATNO','EVENT_ID',cat]]
    # print("score_data shape",score_data.shape) #score_data shape (3445, 3)
    time_list=["V04", "V06", "V08", "V10","V12"]
    five_yeas=pd.DataFrame()
    five_yeas["PATNO"]=score_data['PATNO'].unique()
    # print("count patno",five_yeas["PATNO"].count())#count patno 683
    for time in time_list:
        time_data=score_data[score_data['EVENT_ID']==time]
        # print("time_data",time_data.shape)
        event_data=time_data.rename(columns={cat:time})
        five_yeas=pd.merge(five_yeas,event_data[['PATNO',time]],on='PATNO',how='outer')
    # five_yeas.to_csv(os.path.join(save_path,"{}_follow_up.csv".format(cat)),index=False)
    # print("five years",five_yeas.shape) #(683, 6)

    df_data = pd.read_csv(cave_result)  #
    data=pd.merge(five_yeas,df_data,on='PATNO',how='inner')
    data=data.drop(['PATNO'],axis=1)
    # data = data.dropna(axis=0)
    print(data.columns.tolist())
    order_list=['V04', 'V06', 'V08', 'V10', 'V12']
    # print("len(order_list)",order_list)
    # palet = sns.color_palette("hls", n_cluster)

    cluster_list=[]#['nc']
    cluster_list.extend(int(i) for i in range(n_cluster))

    # data_nc=data[data["cluster_label"] == cluster_list[0]]
    data_1 = data[data["cluster"] == cluster_list[0]]
    data_2 = data[data["cluster"] == cluster_list[1]]
    # data_3_1 = data[(data["cluster_label"] == cluster_list[3]) | (data["cluster_label"] == cluster_list[1])]
    #
    # data_nc = data_nc[order_list].dropna(axis=0)
    # data_1 = data_1[order_list].dropna(axis=0)
    # data_2 = data_2[order_list].dropna(axis=0)
    # data_3_1 = data_3_1[order_list].dropna(axis=0)
    data_1 = data_1.drop('cluster', axis=1)
    data_2 = data_2.drop('cluster', axis=1)

    #
    # n_nc=data_nc.shape[0]
    n_1 = data_1.shape[0]
    n_2 = data_2.shape[0]
    # n_3_1 = data_3_1.shape[0]
    labels=data_1.columns.tolist() # ['0', '3', '6', '9', '12', '18', '24', '30', '36', '42', '48', '54', '60']

    # data_nc.loc['mean']=data_nc.apply(np.mean)
    # data_nc.loc['std']=data_nc.apply(np.std)
    data_1.loc['mean']=data_1.apply(np.mean)
    data_1.loc['std']=data_1.apply(np.std)
    data_2.loc['mean']=data_2.apply(np.mean)
    data_2.loc['std']=data_2.apply(np.std)
    # data_3_1.loc['mean'] = data_3_1.apply(np.mean)
    # data_3_1.loc['std'] = data_3_1.apply(np.std)

    # plt.errorbar(np.arange(len(labels)),data_nc.loc['mean'],data_nc.loc['std'],marker='s', mec='green', ms=20, mew=4) mec='green' mec='blue',
    plt.errorbar(np.arange(len(labels)), round(data_1.loc['mean'],2), round(data_1.loc['std']/np.sqrt(n_1),2), marker='s', ms=7, mew=1,elinewidth=1,capsize=3,label='Subgroup 1')
    plt.errorbar(np.arange(len(labels)), round(data_2.loc['mean'],2), round(data_2.loc['std']/np.sqrt(n_2),2), marker='s',  ms=7, mew=1,elinewidth=1,capsize=3,label='Subgroup 2')
    # plt.errorbar(np.arange(len(order_list)), data_3_1.loc['mean'], data_3_1.loc['std'] / np.sqrt(n_3_1), marker='s',
    #              ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 3_1')
    if n_cluster>2:
        data_3 = data[data["cluster"] == cluster_list[2]]
        # data_3=data_3[order_list].dropna(axis=0)
        data_3 = data_3.drop('cluster', axis=1)
        n_3 = data_3.shape[0]
        data_3.loc['mean'] = data_3.apply(np.mean)
        data_3.loc['std'] = data_3.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_3.loc['mean'],2), round(data_3.loc['std']/np.sqrt(n_3),2), marker='s', ms=7,
                     mew=1,elinewidth=1,capsize=3,label='pattern 3')
        # print(data_1.shape, data_2.shape, data_nc.shape,data_3.shape)
    if n_cluster>3:
        data_4 = data[data["cluster"] == cluster_list[3]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_4 = data_4.drop('cluster', axis=1)
        n_4 = data_4.shape[0]
        data_4.loc['mean'] = data_4.apply(np.mean)
        data_4.loc['std'] = data_4.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_4.loc['mean'], 2), round(data_4.loc['std'] / np.sqrt(n_4), 2),
                     marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 4')
    if n_cluster>4:
        data_5 = data[data["cluster"] == cluster_list[4]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_5 = data_5.drop('cluster', axis=1)
        n_5 = data_5.shape[0]
        data_5.loc['mean'] = data_5.apply(np.mean)
        data_5.loc['std'] = data_5.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_5.loc['mean'], 2), round(data_5.loc['std'] / np.sqrt(n_5), 2),
                     marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 4')
    plt.legend()
    plt.xticks(range(len(labels)),labels)
    plt.xlabel("Follow-up time(months)")
    plt.ylabel(cat)
    # plt.show()
    plt.savefig(os.path.join(save_path,"{}_followup_pattern.png".format(cat)))
    plt.close()

def follow_errorbar(cave_result,follow_data,n_cluster,save_path,cat):
    df_data = pd.read_csv(cave_result)
    df_data.drop_duplicates('PATNO',keep='first',inplace=True)
    df_data=df_data[['PATNO','cluster']]
    score = pd.read_csv(follow_data)
    # print(score.head())
    # score["cluster"] = df_data
    data=pd.merge(score,df_data,on='PATNO',how='inner')
    print(data.columns.tolist())
    # data = data.drop(["age",'gen',"PATNO"],axis=1)
    order_list = ['0', '3', '6', '9', '12', '18', "24", '30', '36', '42', '48', '54', '60']
    data = data.drop(["PATNO",'72','84','96'], axis=1)
    # data=data.dropna(axis=0)
    # data.rename(columns={'X0':'0', 'X3':'3', 'X6':'6', 'X9':'9', 'X12':'12', 'X18':"18", 'X24':'24', 'X30':'30', 'X36':'36', 'X42':'42', 'X48':'48', 'X54':"54",'X60':'60'},inplace=True)
    colors = ["#9b59b6", "#2ecc71", "#3498db", "#e74c3c", "#95a5a6", "#34495e"]
    cluster_list=[]#['nc']
    cluster_list.extend(int(i) for i in range(n_cluster))
    # print(cluster_list)
    # print(data.columns.tolist())
    # b = data["cluster"].unique()
    # print(b)
    # a=list(b)
    # cluster_list.extend(a)
    # # print(a)
    print(cluster_list)
    print(data['cluster'].unique())
    # data_nc=data[data["cluster_label"] == cluster_list[0]]
    data_1 = data[data["cluster"] == cluster_list[0]]
    data_2 = data[data["cluster"] == cluster_list[1]]
    # print(data_1.columns.to_list)

    # data_3_1=data[(data["cluster_label"]==cluster_list[3]) | (data["cluster_label"] == cluster_list[1])]

    # data_nc = data_nc[order_list].dropna(axis=0)
    # data_1=data_1[order_list].dropna(axis=0)
    # data_2 = data_2[order_list].dropna(axis=0)

    print(data_1.shape, data_2.shape)

    # data_nc.loc['mean'] = data_nc.apply(np.mean)
    # data_nc.loc['std'] = data_nc.apply(np.std)
    data_1=data_1.drop('cluster',axis=1)
    data_1.loc['mean'] = data_1.apply(np.mean)
    data_1.loc['std'] = data_1.apply(np.std)
    data_2 = data_2.drop('cluster',axis=1)
    data_2.loc['mean'] = data_2.apply(np.mean)
    data_2.loc['std'] = data_2.apply(np.std)

    # data_3_1.loc['mean'] = data_3_1.apply(np.mean)
    # data_3_1.loc['std'] = data_3_1.apply(np.std)
    n_1 = data_1.shape[0]
    n_2 = data_2.shape[0]

    plt.errorbar(np.arange(len(order_list)), data_1.loc['mean'], data_1.loc['std'] / np.sqrt(n_1), marker='s', ms=7, mew=1,
                 elinewidth=1, capsize=3, label='Subgroup 1')
    plt.errorbar(np.arange(len(order_list)), data_2.loc['mean'], data_2.loc['std'] / np.sqrt(n_2), marker='s', ms=7, mew=1,
                 elinewidth=1, capsize=3, label='Subgroup 2')
    if n_cluster>2:
        data_3 = data[data["cluster"] == cluster_list[2]]
        # print(data_3.columns.to_list())
        # data_3 = data_3[order_list].dropna(axis=0)
        data_3 = data_3.drop('cluster',axis=1)
        data_3.loc['mean'] = data_3.apply(np.mean)
        data_3.loc['std'] = data_3.apply(np.std)
        n_3 = data_3.shape[0]
        print(data_3.shape)
        plt.errorbar(np.arange(len(order_list)), data_3.loc['mean'], data_3.loc['std'] / np.sqrt(n_3), marker='s', ms=7,
                     mew=1, elinewidth=1, capsize=3, label='pattern 3')
    if n_cluster>3:
        data_4 = data[data["cluster"] == cluster_list[3]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_4 = data_4.drop('cluster',axis=1)
        data_4.loc['mean'] = data_4.apply(np.mean)
        data_4.loc['std'] = data_4.apply(np.std)
        n_4 = data_4.shape[0]
        print(data_4.shape)
        plt.errorbar(np.arange(len(order_list)), data_4.loc['mean'], data_4.loc['std'] / np.sqrt(n_4), marker='s', ms=7,
                     mew=1, elinewidth=1, capsize=3, label='pattern 4')
    # plt.errorbar(np.arange(len(order_list)), data_3_1.loc['mean'], data_3_1.loc['std'] / np.sqrt(n_3_1), marker='s', ms=7,
    #              mew=1, elinewidth=1, capsize=3, label='pattern 3_1')
    if n_cluster>4:
        data_5 = data[data["cluster"] == cluster_list[4]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_5 = data_5.drop('cluster', axis=1)
        n_5 = data_5.shape[0]
        data_5.loc['mean'] = data_5.apply(np.mean)
        data_5.loc['std'] = data_5.apply(np.std)
        plt.errorbar(np.arange(len(order_list)), round(data_5.loc['mean'], 2), round(data_5.loc['std'] / np.sqrt(n_5), 2),
                     marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 4')

    plt.legend()
    plt.xticks(range(len(order_list)), order_list)
    plt.xlabel("Follow-up time(months)")
    plt.ylabel(cat)
    # plt.show()
    plt.savefig(os.path.join(save_path, "{}_followup_pattern.png".format(cat)),dpi=300)
    plt.close()


def follow_errorbar_v(cave_result, follow, n_cluster, cat,save_path):
    df_data = pd.read_csv(cave_result)
    df_data.drop_duplicates('PATNO', keep='first', inplace=True)
    df_data=df_data[['PATNO','cluster']]
    # df_data=cave_result
    follow_up_i=pd.read_csv(follow)
    follow_up_i.drop(['V14','V15','V17'],axis=1,inplace=True)
    if cat in ['pigd','tremor','seadl']:
        follow_up_i.drop(['V13','V16','V18'],axis=1,inplace=True)
    data=pd.merge(df_data,follow_up_i,on='PATNO',how='inner')
    data = data.drop(['PATNO'], axis=1)
    # print(data.shape)
    # data=data.dropna(axis=0)
    # print(data.shape)
    cluster_list = []  # ['nc']
    cluster_list.extend(int(i) for i in range(n_cluster))
    data_1 = data[data["cluster"] == cluster_list[0]]
    data_2 = data[data["cluster"] == cluster_list[1]]
    data_1 = data_1.drop('cluster', axis=1)
    data_2 = data_2.drop('cluster', axis=1)

    #
    # n_nc=data_nc.shape[0]
    n_1 = data_1.shape[0]
    n_2 = data_2.shape[0]
    # n_3_1 = data_3_1.shape[0]
    labels = data_1.columns.tolist()
    if len(labels) >6:
        x_ticks=['0', '3', '6', '9', '12', '18', '24', '30', '36', '42', '48', '54', '60']
    else:
        x_ticks=['0', '12', '24', '36', '48', '60']
    # data_nc.loc['mean']=data_nc.apply(np.mean)
    # data_nc.loc['std']=data_nc.apply(np.std)
    data_1.loc['mean'] = data_1.apply(np.mean)
    data_1.loc['std'] = data_1.apply(np.std)
    data_2.loc['mean'] = data_2.apply(np.mean)
    data_2.loc['std'] = data_2.apply(np.std)
    # data_3_1.loc['mean'] = data_3_1.apply(np.mean)
    # data_3_1.loc['std'] = data_3_1.apply(np.std)

    # plt.errorbar(np.arange(len(labels)),data_nc.loc['mean'],data_nc.loc['std'],marker='s', mec='green', ms=20, mew=4) mec='green' mec='blue',
    plt.errorbar(np.arange(len(labels)), round(data_1.loc['mean'], 2), round(data_1.loc['std'] / np.sqrt(n_1), 2),
                 marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='Subgroup 1')
    plt.errorbar(np.arange(len(labels)), round(data_2.loc['mean'], 2), round(data_2.loc['std'] / np.sqrt(n_2), 2),
                 marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='Subgroup 2')
    # plt.errorbar(np.arange(len(order_list)), data_3_1.loc['mean'], data_3_1.loc['std'] / np.sqrt(n_3_1), marker='s',
    #              ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 3_1')

    if n_cluster > 2:
        data_3 = data[data["cluster"] == cluster_list[2]]
        # data_3=data_3[order_list].dropna(axis=0)
        data_3 = data_3.drop('cluster', axis=1)
        n_3 = data_3.shape[0]
        data_3.loc['mean'] = data_3.apply(np.mean)
        data_3.loc['std'] = data_3.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_3.loc['mean'], 2), round(data_3.loc['std'] / np.sqrt(n_3), 2),
                     marker='s', ms=7,
                     mew=1, elinewidth=1, capsize=3, label='pattern 3')
        # print(data_1.shape, data_2.shape, data_nc.shape,data_3.shape)
    if n_cluster > 3:
        data_4 = data[data["cluster"] == cluster_list[3]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_4 = data_4.drop('cluster', axis=1)
        n_4 = data_4.shape[0]
        data_4.loc['mean'] = data_4.apply(np.mean)
        data_4.loc['std'] = data_4.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_4.loc['mean'], 2), round(data_4.loc['std'] / np.sqrt(n_4), 2),
                     marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 4')
    if n_cluster > 4:
        data_5 = data[data["cluster"] == cluster_list[4]]
        # data_4 = data_4[order_list].dropna(axis=0)
        data_5 = data_5.drop('cluster', axis=1)
        n_5 = data_5.shape[0]
        data_5.loc['mean'] = data_5.apply(np.mean)
        data_5.loc['std'] = data_5.apply(np.std)
        plt.errorbar(np.arange(len(labels)), round(data_5.loc['mean'], 2), round(data_5.loc['std'] / np.sqrt(n_5), 2),
                     marker='s', ms=7, mew=1, elinewidth=1, capsize=3, label='pattern 4')

    plt.legend()
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.xlabel("Follow-up time(months)")
    plt.ylabel(cat)
    # plt.show()
    plt.savefig(os.path.join(save_path, "{}_followup_pattern_2.png".format(cat)),dpi=300)
    plt.close()

def draw_linear_fit(cave_result,follow,cat):
    df_data = pd.read_csv(cave_result)
    print(df_data.shape)
    follow_up_i = pd.read_csv(follow)
    follow_up_i.drop(['V14', 'V15', 'V17'], axis=1, inplace=True)
    if cat in ['pigd', 'tremor', 'seadl']:
        follow_up_i.drop(['V13', 'V16', 'V18'], axis=1, inplace=True)
    data = pd.merge(df_data, follow_up_i, on='PATNO', how='inner')
    # ori_data=pd.read_csv(ori)

    print(data.shape)
    print(data.head())

    data.set_index(keys=['PATNO','cluster'],inplace=True)
    data.columns.names = ['stage']
    data_stack=data.stack()
    print(data_stack.head())
    # # print(data_stack.columns.tolist())
    # # print(data_stack.index[0])
    data_stack=pd.DataFrame(data_stack,columns=['values'])
    # print(data_stack.shape)
    # print(data_stack.head())
    data_stack.to_csv("/home/istbi/HD1/chengyu2_dataset/PPMI/updrs_follow/{}_stack.csv".format(cat))
    # data=pd.read_csv("/home/istbi/HD1/chengyu2_dataset/PPMI/updrs_follow/moca_stack.csv")
    # data['month']=pd.DataFrame(np.array([int(i[1:]) if "V" in i else 1 for i in data['stage'].values.tolist() ] ))
    # print(data.head())
    # # data=data[data['cluster']==1]
    # sns.set_style("white")#sns.lmplot hue="cluster",
    # # gridobj = sns.regplot(x="month", y="values",  data=data,
    # #                      height=7, aspect=1.6, robust=True, palette='tab10',
    # #                      scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
    # gridobj =sns.lmplot(x="month", y="values", data=data, hue="cluster", x_jitter=.05)#,x_estimator=np.mean
    # # gridobj =sns.regplot(x="month", y="values", data=data)
    # # Decorations
    # gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
    # plt.title("{} best fit grouped by cluster".format(cat), fontsize=10)
    # plt.show()

def linear_mixed_effected_model_variables(cave_result,follow,ori,cat,save_path):#path,cat
    # data=pd.read_csv(path)
    df_data = pd.read_csv(cave_result)
    print(df_data.shape)
    follow_up_i = pd.read_csv(follow)
    follow_up_i.drop(['V14', 'V15', 'V17'], axis=1, inplace=True)
    if cat in ['pigd', 'tremor', 'seadl']:
        follow_up_i.drop(['V13', 'V16', 'V18'], axis=1, inplace=True)

    data = pd.merge(df_data, follow_up_i, on='PATNO', how='inner')
    data.to_csv(os.path.join(save_path,'{}_merge_cluster_nostack.csv'.format(cat)),index=False)
    data.set_index(keys=['PATNO', 'cluster'], inplace=True)
    data.columns.names = ['stage']
    data_stack = data.stack()
    print(data_stack.head())
    # # print(data_stack.columns.tolist())
    # # print(data_stack.index[0])
    data_stack = pd.DataFrame(data_stack, columns=['values'])
    data_stack.to_csv(os.path.join(save_path, "{}_stack.csv".format(cat)))

    data = pd.read_csv(os.path.join(save_path,"{}_stack.csv".format(cat)))
    data['month'] = pd.DataFrame(np.array([int(i[1:]) if "V" in i else 1 for i in data['stage'].values.tolist()]))
    dic={"BL":0,"V04":12,'V06':24,'V08':36,'V10':48,'V12':60}
    data['month']=data['stage'].map(dic)
    # print(data.shape)

    ori_data=pd.read_csv(ori)
    ori_data = ori_data[(ori_data['EVENT_ID'] == 'BL') & (ori_data['APPRDX'] != 3)]
    variable=['PATNO','age','gen','EDUCYRS',"race",'SITE']
    ori_data=ori_data[variable]
    # print(ori_data.shape)

    data=pd.merge(data,ori_data, on='PATNO', how='inner')
    # print(data.shape)
    # print(data.head())
    data['gen']=data['gen'].apply(lambda x:x-1)
    # print(data.head())
    data=data.dropna(axis=0)
    # data
    # print(data.shape)
    data.to_csv(os.path.join(save_path,"{}_stack_all.csv".format(cat)),index=False)
    #
    # data = pd.read_csv("/home/istbi/HD1/chengyu2_dataset/PPMI/updrs_follow/moca_stack.csv")
    # data['month'] = pd.DataFrame(np.array([int(i[1:]) if "V" in i else 1 for i in data['stage'].values.tolist()]))
    # # print(data.head())
    #
    # # data = sm.datasets.get_rdataset("dietox", "geepack").data
    # # print(data.head())
    # md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"]),re_formula="~cluster+month"

    # md = smf.mixedlm("values ~ EDUCYRS+age+SITE+race+gen+ C(cluster)+ C(cluster):C(month)", data, groups=data['cluster'])
    # model = sm.MixedLM.from_formula(
    #     "values ~ month+age+gen", data, re_formula="0 + month", groups=data["PATNO"])

    # mdf = md.fit()
    # model_fit=model.fit()
    # print(mdf.random_effects)
    print("--------------*******************")


def merge_data_info_subgroup(sustain_cluster,keys):

    # df = pd.read_csv('/home/istbi/HD1/chengyu2_dataset/zheda_t1/zheda_t1_bet_MNI_BL_clinicaldata.csv')
    # df.rename(columns={'性别': 'Sex', '年龄': 'Age', '受教育年限': 'Education', 'II ': 'UPDRS II', 'III': 'UPDRS III',
    #                    'IV': 'UPDRS IV', '总分': 'UPDRS total',
    #                    'RBDQ-HK总分': 'RBDQ-HK total'}, inplace=True)
    # patients = df['APPRDX'].values == "PD"


    variable_csv = '/home/istbi/HD1/chengyu2_dataset/PPMI/cvae/cat12_dbm/ppmi_IQR_BL_name_id_variable_32.csv'
    df = pd.read_csv(variable_csv)
    patients = df['APPRDX'].values == 1  # "PD"
    nc=df['APPRDX'].values==2
    df_asd = df.iloc[patients]
    df_nc=df.iloc[nc]

    cluster=pd.read_csv(sustain_cluster)
    # cluster.rename(columns={'subj_id':"PATNO",'ml_subtype':'cluster'},inplace=True)
    print(cluster.shape)
    df_asd=pd.merge(cluster[['PATNO','cluster']],df_asd,on='PATNO',how='inner')
    print(df_asd.shape)
    cat=['PATNO','cluster']
    cat.extend(keys)
    df_asd=df_asd[cat]
    print(df_asd.shape)
    # data = np.load('../Data/ppmi_fast_freesurfer_pd_nc.npy')
    print(df_asd.columns.tolist())
    return patients,df_asd,df_nc

def prepare_lmer_input_updrs(cave_result,follow,cat,ori,save_path):
    df_data = pd.read_csv(cave_result)
    print(df_data.shape)
    follow_up_i = pd.read_csv(follow)
    # follow_up_i.drop(['V13','V14', 'V15', 'V17','V16', 'V18'], axis=1, inplace=True)
    columes=['PATNO','BL','V01','V02', 'V03','V04',  'V05','V06','V07', 'V08', 'V09','V10', 'V11','V12']#'V13',,'V14','V15'
    columes_moca=['PATNO','V04','V06', 'V08','V10', 'V12']#,'V14','V15'
    # if cat in ['pigd', 'tremor', 'seadl']:
    #     follow_up_i.drop(['V13', 'V16', 'V18'], axis=1, inplace=True)
    if cat=='moca':
        columes=columes_moca
    follow_up_i=follow_up_i[columes]
    data = pd.merge(df_data, follow_up_i, on='PATNO', how='inner')
    # ori_data=pd.read_csv(ori)

    print(data.shape)
    print(data.head())

    data.set_index(keys=['PATNO', 'cluster'], inplace=True)
    data.columns.names = ['stage']
    data_stack = data.stack()
    print(data_stack.head())
    # # print(data_stack.columns.tolist())
    # # print(data_stack.index[0])
    data_stack = pd.DataFrame(data_stack, columns=['values'])
    data_stack.to_csv(os.path.join(save_path,"{}_stack.csv".format(cat)))#先保存再读取才有stage
    data=pd.read_csv(os.path.join(save_path,"{}_stack.csv".format(cat)))
    data['month'] = pd.DataFrame(np.array([int(i[1:]) if "V" in i else 1 for i in data['stage'].values.tolist()]))
    dic = {"BL": 0,'V01':3,'V02':6,'V03':9, "V04": 12, "V05":18,'V06': 24,'V07': 30,  'V08': 36,'V09': 42, 'V10': 48, 'V11': 54, 'V12': 60}#'V13':72,V14':84,'V15':96
    data['month'] = data['stage'].map(dic)
    # print(data.shape)

    ori_data = pd.read_csv(ori)
    ori_data = ori_data[(ori_data['EVENT_ID'] == 'BL') & (ori_data['APPRDX'] != 3)]
    variable = ['PATNO', 'age', 'gen', 'EDUCYRS', "race", 'SITE']
    ori_data = ori_data[variable]
    # print(ori_data.shape)

    data = pd.merge(data, ori_data, on='PATNO', how='inner')
    # print(data.shape)
    # print(data.head())
    data['gen'] = data['gen'].apply(lambda x: x - 1)
    # print(data.head())
    data = data.dropna(axis=0)
    # print(data.shape)
    data.to_csv(os.path.join(save_path,"{}_stack_all.csv".format(cat)), index=False)
def select_sujects_60_month():
    varibale_path='/home/daiyx/zhenglp/model/CVAE/ppmi_zheda_bspline_MNI/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_5.15/pca_follow_varia_stack_subgroup/moca_stack_all.csv'
    data=pd.read_csv(varibale_path)
    count_data=pd.DataFrame(np.array(data['PATNO'].value_counts().index),columns=['PATNO'])
    count_data['count'] = pd.DataFrame(np.array(data['PATNO'].value_counts()))
    count_data=count_data[count_data['count']==6]
    print(count_data.shape)
    select_variable=pd.merge(data,count_data['PATNO'],on='PATNO',how='inner')
    print(select_variable.shape)
    print(select_variable.head())
    select_variable.to_csv(varibale_path.replace('.csv','60_month.csv'),index=False)

def renew_Moca_data(ori):
    save_path_moca='/home/daiyx/zhenglp/model/CVAE/ppmi_zheda_bspline_MNI/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_5.15/pca_follow_varia_stack_subgroup'
    cluster_result=pd.read_csv("/home/daiyx/zhenglp/model/CVAE/ppmi_zheda_bspline_MNI/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_5.15/Specific_vec_pd_pca_PATNO_cluster_kmeans.csv")
    new_moca=pd.read_csv("/home/daiyx/zhenglp/dataset/PPMI/PPMI/ppmi_score/Montreal_Cognitive_Assessment__MoCA_.csv")
    new_moca=new_moca[['PATNO','EVENT_ID','MCATOT']]
    time_list=["SC","V04","V06","V08","V10","V12",'V13','V14','V15']
    new_moca=new_moca[new_moca.EVENT_ID.isin(time_list)]
    print(new_moca.head(10))
    new_moca_cluster=pd.merge(new_moca,cluster_result,on='PATNO',how='inner')
    new_moca_cluster.rename(columns={'EVENT_ID':'stage','MCATOT':'values'},inplace=True)
    dic = {"SC": 0, "V04": 12,  'V06': 24,  'V08': 36, 'V10': 48, 'V12': 60, 'V13':72,'V14':84,'V15':96}  # 'V13':72,V14':84,'V15':96
    new_moca_cluster['month'] =new_moca_cluster['stage'].map(dic)
    print(new_moca_cluster.head(10))
    # index=new_moca_cluster[new_moca_cluster['PATNO']==3173].index
    # new_moca_cluster.drop(index,axis=0,inplace=True)
    # index2=new_moca_cluster[new_moca_cluster['PATNO']==4083].index
    # new_moca_cluster.drop(index2,axis=0,inplace=True)
    new_moca_cluster.to_csv(os.path.join(save_path_moca,"new_moca_stack_8y.csv"),index=False)
    ori_data = pd.read_csv(ori)
    ori_data = ori_data[(ori_data['EVENT_ID'] == 'BL') & (ori_data['APPRDX'] != 3)]
    variable = ['PATNO', 'age', 'gen', 'EDUCYRS', "race", 'SITE']
    ori_data = ori_data[variable]
    # print(ori_data.shape)

    data = pd.merge(new_moca_cluster, ori_data, on='PATNO', how='inner')
    # print(data.shape)
    # print(data.head())
    data['gen'] = data['gen'].apply(lambda x: x - 1)

    data.to_csv(os.path.join(save_path_moca,"new_moca_stack_all.csv"),index=False)

    count_data = pd.DataFrame(np.array(data['PATNO'].value_counts().index), columns=['PATNO'])
    count_data['count'] = pd.DataFrame(np.array(data['PATNO'].value_counts()))
    count_data = count_data[count_data['count'] == 6]
    select_variable = pd.merge(data, count_data['PATNO'], on='PATNO', how='inner')
    print(select_variable.shape)
    print(select_variable.head())
    select_variable.to_csv(os.path.join(save_path_moca,"new_moca_stack_8y_all.csv"),index=False)

def slop_2_subgroup_barplot():
    save_path='/home/daiyx/zhenglp/model/CVAE/ppmi_zheda_bspline_MNI/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_5.15'
    cluster=pd.read_csv(os.path.join(save_path,'Specific_vec_pd_umap_PATNO_cluster_kmeans.csv'))
    slop_interp=pd.read_csv(
        "/home/daiyx/zhenglp/dataset/PPMI/cvae/linear_mixed_effect_model_slop/5.15_umap/moca_updrs123_slop_interp.csv",
        )
    # slop_inter=slop_interp[slop_interp['APPRDX']]
    print(slop_interp.shape)
    slop_interp=slop_interp[['MoCA_slop','PATNO']]
    slop_interp_cluster=pd.merge(cluster,slop_interp,on='PATNO',how='inner')
    slop_interp_cluster['cluster']=slop_interp_cluster['cluster'].apply(lambda x:x+1)
    print(slop_interp_cluster['cluster'].count())
    delate=slop_interp_cluster[slop_interp_cluster['MoCA_slop']<-0.14]
    print(delate['PATNO'])
    # "id=["3116",'3700','3771','3818','4114']"

    print(slop_interp_cluster.groupby('cluster').MoCA_slop.describe())

    # sns.scatterplot('PATNO','MoCA_slop',data=slop_interp_cluster,hue='cluster')

    sns.boxplot('cluster','MoCA_slop',data=slop_interp_cluster)
    plt.show()
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认box_plot(scale=3)进行清洗
    param data: 接收pandas数据格式
    param col_name: pandas列名
    param scale: 尺度
    """
    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_serier = data_n[col_name]
    rule, value = box_plot_outliers(data_serier, box_scale=scale)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]
    print("Delete number is:{}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is:{}".format(data_n.shape[0]))
    index_low = np.arange(data_serier.shape[0])[rule[0]]
    outliers = data_serier.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_serier.shape[0])[rule[1]]
    outliers = data_serier.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


if __name__ == '__main__':
    save_path='./step4_result_739'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    default_keys = ["SITE", "age", "gen", "updrs3_score", "rem", "scopa", "updrs1_score",
                    "updrs2_score", "moca", "updrs_totscore", 'TESTVALUE']  # 'ICD',# 'ICD',
    default_keys_t_p = [ "age", "gen", "rem", "scopa", "moca", "updrs1_score",
                    "updrs2_score", "updrs3_score", "updrs_totscore", 'serum_NFL', "CAUDATE_R",
                    "CAUDATE_L", 'info-sxdt', 'info-pddxdt', "asyn", "tau", "ptau", "abeta"] #"SITE",

    npz_path='./step2_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001.npz'
    n_samples=10
    name=npz_path.split('/')[-1].split('.npz')[0]
    save_path=os.path.join(save_path,name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    cluster_save_path=os.path.join(save_path,'cluster_result_kmeans_11.29')
    if os.path.exists(cluster_save_path) is False:
        os.mkdir(cluster_save_path)

    data,patients,patients_id=load_data(npz_path)
    # arr_sl, arr_bg, arr_vae, n=get_feature(data, cnvs, patients)
    arr_sl, arr_bg, arr_vae, n = get_feature_one_dataset(data, patients,n_samples)
    plot_clustering_result(arr_sl, arr_bg, arr_vae, n,cluster_save_path)
    umap_reduce(data,patients,patients_id,cluster_save_path)


    cave_result=os.path.join(save_path,'Specific_vec_pd_PATNO_cluster_kmeans.csv')
    stastic_result=os.path.join(save_path,'stastic_result_kmeans')
    if os.path.exists(stastic_result) is False:
        os.mkdir(stastic_result)
    ori_score_data = '../PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv'
    # # ori_score_data='/home/istbi/HD1/chengyu2_dataset/zheda_t1/zheda_t1_BL_clinicaldata_pd_192_nc.csv'
    data=pd.read_csv(ori_score_data)
    data.rename(columns={'性别':'Sex', '年龄':'Age', '受教育年限':'Education','II ':'UPDRS II', 'III':'UPDRS III', 'IV':'UPDRS IV', '总分':'UPDRS total',
                       'RBDQ-HK总分':'RBDQ-HK total'},inplace=True)
    downscaling='pca'
    cave_result=os.path.join(cluster_save_path,'Specific_vec_pd_{}_PATNO_cluster_kmeans.csv'.format(downscaling))
    follow_result=os.path.join(cluster_save_path,'follow_up_result_{}_kmeans'.format(downscaling))
    if os.path.exists(follow_result) is False:
        os.mkdir(follow_result)

    # moca_slop_subgroup = "/home/istbi/HD1/chengyu2_dataset/PPMI/updrs_follow/moca_stack_all_pd_latent_interp_sloper_subgroup.csv"
    # latent_subgroup="/home/istbi/HD1/chengyu2_dataset/PPMI/updrs_follow/moca_stack_all_pd_latent_subgroup.csv"
    cat_list = ["total_MDS-UPDRS", "MDS-UPDRS-Part_II", "MDS-UPDRS-Part_I", "MDS-UPDRS-Part_III", "PIGD_score",
                "Schwab_England_ADL" ]
    updrs2='/mnt/raid/chengyu2_dataset/PPMI/updrs_score2'










