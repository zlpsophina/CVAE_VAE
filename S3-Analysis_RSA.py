import os

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib import font_manager
from statsmodels.stats.multitest import multipletests

import scipy
from sklearn.decomposition import PCA
from helper_funcs import *
from scipy import stats as stats
import scipy.io as scio
#import rsatoolbox
    # %%
def data2cmat(data):
    return np.array([squareform(pdist(data[s,:,:],metric='euclidean')) for s in range(data.shape[0])])

def plot_nice_bar(key, rsa, ax=None, figsize=None, dpi=None, fontsize=None, fontsize_star=None, fontweight=None,
                  line_width=None, marker_size=None, title=None, report_t=False, do_pairwise_stars=False,
                  do_one_sample_stars=True):
    import seaborn as sns
    from scipy.stats import ttest_1samp
    from scipy.stats import ttest_ind as ttest

    pallete = sns.color_palette()
    pallete_new = sns.color_palette()

    if not figsize:
        figsize = (5, 2)
    if not dpi:
        dpi = 300

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)



    data = rsa[key]
    # print(key)
    # print("data",data.shape) #data (10, 3)
    n = data.shape[0]
    c = data.shape[1]
    x = np.arange(c)

    if not fontsize:
        fontsize = 16

    if not fontsize_star:
        fontsize_star = 25
    if not fontweight:
        fontweight = 'bold'
    if not line_width:
        line_width = 2.5
    if not marker_size:
        marker_size = .1

    for i in range(c):
        plot_data = np.zeros(data.shape)
        plot_data[:, i] = data[:, i]

        xs = np.repeat(i, n) + (np.random.rand(n) - .5) * .25
        sc = plt.scatter(xs, data[:, i], c='k', s=marker_size)
        b = sns.barplot(data=plot_data, errcolor='r', linewidth=line_width, errwidth=line_width,
                        facecolor=np.hstack((np.array(pallete_new[i]), .3)),
                        edgecolor=np.hstack((np.array(pallete_new[i]), 1)))
        # b = sns.barplot(data=plot_data, errcolor='r', linewidth=line_width, errwidth=line_width,
        #                 facecolor=np.hstack((np.array(pallete_new[i]), .3)),
        #                 edgecolor=np.hstack((np.array(pallete_new[i]), 1)))

    locs, labels = plt.yticks()
    new_y = locs
    new_y = np.linspace(locs[0], locs[-1], 6)
    plt.yticks(new_y, labels=[f'{yy:.2f}' for yy in new_y], fontsize=fontsize, fontweight=fontweight)
    plt.ylabel('model fit (r)', fontsize=fontsize, fontweight=fontweight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)

    # xlbls = ['VAE', 'BG', 'SL']
    xlbls=['VAE','shared','Specific']
    plt.xticks(np.arange(3), labels=xlbls, fontsize=fontsize, fontweight=fontweight)

    if do_one_sample_stars:
        one_sample = np.array([ttest_1samp(data[:, i], 0) for i in range(3)])
        one_sample_thresh = np.array((1, .05, .001, .0001))
        one_sample_stars = np.array(('n.s.', '*', '**', '***'))
        xlbls = ['VAE', 'Shared', 'Specific']
        for i in range(c):
            these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
            xlbls[i] = f'{xlbls[i]}\n({these_stars})'
        plt.xticks(np.arange(3), labels=xlbls, fontsize=fontsize, fontweight=fontweight, horizontalalignment='center',
                   multialignment='center')

        #
        # one_sample = np.array([ttest_1samp(data[:, i], 0) for i in range(3)])
        # one_sample_thresh = np.array((1, .05, .001, .0001))
        # one_sample_stars = np.array(('n.s.', '*', '**', '***'))
        # xlbls = ['VAE','Shared','Specific']
        # # print("c",c)
        # for i in range(c):
        #     # print("star",np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])
        #     # these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
        #     these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
        #     xlbls[i] = f'{xlbls[i]}\n({these_stars})'
        # plt.xticks(np.arange(3), labels=xlbls, fontsize=fontsize, fontweight=fontweight, horizontalalignment='center',
        #            multialignment='center')

    pairwise_t = np.zeros((3, 3))
    pairwise_p = np.zeros((3, 3))

    pairwise_sample_thresh = np.array((1, .05, .001, .0001))
    pairwise_sample_stars = np.array(('n.s.', '*', '**', '***'))

    if report_t:
        for i in range(c):
            for j in range(c):
                t, p = ttest(data[:, i], data[:, j])
                mnames = ['VAE','Shared','Specific']

                if p > .001:
                    print(f'{key} {mnames[i]} >  {mnames[j]} | t({data.shape[0] - 1}) = {t:.2f} p = {p:.2f}')
                else:
                    print(f'{key} {mnames[i]} >  {mnames[j]} | t({data.shape[0] - 1}) = {t:.2f} p $<$ .001')
                pairwise_t[i, j] = t
                pairwise_p[i, j] = p

    comps = [[1, 2]]
    if do_pairwise_stars:
        for comp_idx in range(len(comps)):
            this_comp = comps[comp_idx]
            sig_idx = max(np.nonzero(pairwise_p[this_comp[0], this_comp[1]] < pairwise_sample_thresh)[0])
            max_y = new_y[-1] + comp_idx * .05
            xs = np.array(this_comp)
            stars = pairwise_sample_stars[sig_idx]
            plt.plot(xs, [max_y, max_y], 'k', linewidth=line_width)
            plt.text(xs.mean(), max_y, stars, fontsize=fontsize_star, horizontalalignment='center',
                     fontweight=fontweight)

    ylim = plt.ylim()
    plt.ylim(np.array(ylim) * (1, 1.1))

    if not title:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
# %%
def plot_nice_bar_figure(key, rsa, ax=None, figsize=None, dpi=None, fontsize=None, fontsize_star=None, fontweight=None,
                  line_width=None, marker_size=None, title=None, report_t=False, do_pairwise_stars=False,
                  do_one_sample_stars=True):
    import seaborn as sns
    from scipy.stats import ttest_1samp
    from scipy.stats import ttest_ind as ttest

    pallete = sns.color_palette()#rocket 'bright''dark'
    pallete_new = sns.color_palette()

    if not figsize:
        figsize = (5, 2)
    if not dpi:
        dpi = 300

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # pallete_new[1] = pallete[1]
    # pallete_new[0] = pallete[2]
    # pallete_new[0] = tuple(np.array((.5, .5, .5)))

    data = rsa[key]
    print(key)
    print("data",data.shape) #data (10, 3)
    # data=
    n = data.shape[0]
    c = data.shape[1]
    x = np.arange(c)

    if not fontsize:
        fontsize = 16

    if not fontsize_star:
        fontsize_star = 25
    if not fontweight:
        fontweight = 'bold'
    if not line_width:
        line_width = 2.5
    if not marker_size:
        marker_size = .1

    for i in range(c): #3 种latent feature
        plot_data = np.zeros(data.shape)
        plot_data[:,i] = data[:, i]
        # print(plot_data.shape) #(10, 3)
        xs = np.repeat(i, n) + (np.random.rand(n) - .5) * .25
        sc = plt.scatter(data[:, i],xs,  c='k', s=marker_size)
        # plot_data=pd.DataFrame(plot_data)
        # plot_data=plot_data.transpose()
        # b = sns.barplot(data=plot_data, errcolor='r', linewidth=line_width, errwidth=line_width,
        #                 facecolor=np.hstack((np.array(pallete_new[i]), .3)),
        #                 edgecolor=np.hstack((np.array(pallete_new[i]), 1)))
        b = sns.barplot(data=plot_data, errcolor='r', linewidth=line_width, errwidth=line_width,
                        facecolor=np.hstack((np.array(pallete_new[i]), .3)),
                        edgecolor=np.hstack((np.array(pallete_new[i]), 1)),orient="h")

    locs, labels = plt.xticks()
    new_x = locs
    # new_x = np.linspace(locs[0], locs[-1], 6)
    # plt.xticks(new_x, labels=[f'{yy:.2f}' for yy in new_x], fontsize=fontsize, fontweight=fontweight)
    # plt.xlabel('model fit (r)', fontsize=fontsize, fontweight=fontweight)
    plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)

    # xlbls = ['VAE', 'BG', 'SL']
    # ylbls=['VAE','shared','Specific']
    # plt.yticks(np.arange(3),labels=ylbls,  fontsize=fontsize, fontweight=fontweight,rotation=90)
    plt.yticks([])
    if do_one_sample_stars:
        one_sample = np.array([ttest_1samp(data[:, i], 0) for i in range(3)])
        one_sample_thresh = np.array((1, .05, .001, .0001))
        one_sample_stars = np.array(('n.s.', '*', '**', '***'))
        ylbls = ['VAE','Shared','Specific']
        print("c",c)
        for i in range(c):
            print("star",np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])
            these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
            # these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
            # ylbls[i] = f'{ylbls[i]}\n({these_stars})'
        #     ylbls[i] = f'({these_stars})'
        # plt.yticks(np.arange(3), labels=ylbls,fontsize=fontsize, fontweight=fontweight, horizontalalignment='center',
        #            multialignment='center')#

    pairwise_t = np.zeros((3, 3))
    pairwise_p = np.zeros((3, 3))

    pairwise_sample_thresh = np.array((1, .05, .001, .0001))
    pairwise_sample_stars = np.array(('n.s.', '*', '**', '***'))

    if report_t:
        for i in range(c):
            for j in range(c):
                t, p = ttest(data[:, i], data[:, j])
                mnames = ['VAE','Shared','Specific']

                if p > .001:
                    print(f'{key} {mnames[i]} >  {mnames[j]} | t({data.shape[0] - 1}) = {t:.2f} p = {p:.2f}')
                else:
                    print(f'{key} {mnames[i]} >  {mnames[j]} | t({data.shape[0] - 1}) = {t:.2f} p $<$ .001')
                pairwise_t[i, j] = t
                pairwise_p[i, j] = p

    comps = [[1, 2]]
    if do_pairwise_stars:
        for comp_idx in range(len(comps)):
            this_comp = comps[comp_idx]
            sig_idx = max(np.nonzero(pairwise_p[this_comp[0], this_comp[1]] < pairwise_sample_thresh)[0])
            max_y = new_x[-1] + comp_idx * .05
            xs = np.array(this_comp)
            stars = pairwise_sample_stars[sig_idx]
            plt.plot(xs, [max_y, max_y], 'k', linewidth=line_width)
            plt.text(xs.mean(), max_y, stars, fontsize=fontsize_star, horizontalalignment='center',
                     fontweight=fontweight)

    # xlim = plt.xlim()
    # plt.xlim(np.array(xlim) * (1, 1.1))

    # if not title:
    #     plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    # else:
    #     plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
def generate_dissimilarity_metrics(npy_path,save_path):
    data_latent_vec = np.load(npy_path)  # Load latent representations
    file_name=npy_path.split('/')[-1].split('.npz')[0]
    print(list(data_latent_vec.keys()))
    # Split dictionary into separate variables
    salient_vec_abide = data_latent_vec['salient_vec_abide'] #10 466 16
    background_vec_abide = data_latent_vec['background_vec_abide']
    vae_vec_abide = data_latent_vec['vae_vec_abide']
    pd_latent=np.mean(salient_vec_abide,axis=0)
    print(pd_latent.shape)
    # scio.savemat(os.path.join(save_path,'{}.mat'.format(file_name)), {'salient_vec_abide':pd_latent}) #,"background_vec_abide":background_vec_abide,"vae_vec_abide":vae_vec_abide


    # salient_vec_abide=salient_vec_abide[:477,:]
    # background_vec_abide=background_vec_abide[:477,:]
    # vae_vec_abide=vae_vec_abide[:477,:]

    cmat_salient_vec_abide = data2cmat(salient_vec_abide)
    cmat_background_vec_abide = data2cmat(background_vec_abide)
    cmat_vae_vec_abide = data2cmat(vae_vec_abide)
    return cmat_salient_vec_abide,cmat_background_vec_abide,cmat_vae_vec_abide



def load_legend_make_model(default_keys,variable_csv,num_csf):
    df = pd.read_csv(variable_csv)
    df.rename(columns={'age': 'Age', 'gen': 'Gen', 'moca': 'MoCA','updrs1_score':'UPDRS I','rem':'REM','scopa':'Scopa',
                       'updrs2_score': 'UPDRS II', 'updrs3_score': 'UPDRS III', 'updrs_totscore': 'UPDRS Total',
                       'Serum_NFL': 'Serum_NfL','duration':'Duration',
                       'info-sxdt': 'Info-SXDT',  "asyn": 'α-Synuclein', "tau":  "T-Tau",
                       "ptau": "P-Tau", "abeta":"A{}42".format(r'$\beta$'),  "mean_putamen":"Mean_Putamen","mean_caudate": "Mean_Caudate","mean_striatum": "Mean_Striatum"
                       ,"EDUCYRS":"Education"}, inplace=True)  # 'updrs1':'UPDRS 1','info-pddxdt': 'Info-PDDXDT',

    # default_keys = ["SITE", "Age", "Gen", 'Serum_NfL', "Asyn", "T-Tau", "P-Tau", "A{}42".format(r'$\beta$'), "UPDRS I",
    #                 "UPDRS II", "UPDRS III", "UPDRS Total", "MoCA", "Mean_Putamen", "Mean_Caudate", "Mean_Striatum"]
    # df.rename(columns={'性别':'Sex', '年龄':'Age', '受教育年限':'Education','II ':'UPDRS II', 'III':'UPDRS III', 'IV':'UPDRS IV', '总分':'UPDRS total',
    #                    'RBDQ-HK总分':'RBDQ-HK total'},inplace=True)
    print(df.columns.tolist())
    patients = df['APPRDX'].values == 1
    controls = df['APPRDX'].values == 2

    plt.figure(figsize=(15, 15))
    # default_keys = ['ADOS_Total', 'ADOS_Social', 'DSMIVTR', 'AgeAtScan', 'Sex', 'ScannerID', 'ScanSiteID', 'FIQ']
    # default_keys = [ "SITE", "age", "gen", "updrs3_score", "rem", "scopa", "updrs1_score",
    #                   "updrs2_score", "moca","updrs_totscore",'TESTVALUE']
    # scales_ = ['ratio', 'ratio', 'ordinal', 'ratio', 'ordinal', 'ordinal', 'ordinal', 'ratio', 'ratio', 'ratio']
    # scales_=['ordinal', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio','ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio',
    #          'ratio', 'ratio', 'ratio','ratio']
    # scales_ = ['ratio', 'ratio', 'ordinal', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio',
    #            'ratio', 'ratio', 'ratio', 'ratio', 'ratio',
    #            'ratio','ratio', 'ratio', 'ratio', 'ratio','ratio']
    # scales_=['ordinal', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio','ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', ] #supplementary
    # scales_=['ratio', 'ratio', 'ratio','ratio']#figure 1
    # scales_=['ratio', 'ratio', 'ordinal', 'ratio', 'ratio', 'ratio', 'ratio',]
    scales_=['ratio' for i in range(num_csf)]

    model_rdms = dict()
    model_idxs = dict()
    num=[]
    for i in range(len(default_keys)):

        inVec = df[default_keys[i]].values[patients] #
        idx = ~np.isnan(inVec)
        inVec = inVec[idx]
        print(default_keys[i],inVec.shape)
        num.append(inVec.shape[0])
        this_rdm = make_RDM(inVec, data_scale=scales_[i])

        model_rdms.update({default_keys[i]: this_rdm})
        model_idxs.update({default_keys[i]: idx})
    return df,model_rdms,model_idxs,num

def fit_model(rsa_results,df,model_rdms,model_idxs,cmat_vae_vec_abide, cmat_background_vec_abide, cmat_salient_vec_abide):

    patients = df['APPRDX'].values == 1

    data = [cmat_vae_vec_abide, cmat_background_vec_abide, cmat_salient_vec_abide]

    # rsa_results = dict()
    # for key in default_keys:
    #     res = np.array([fit_rsa(datum, key,model_idxs,model_rdms,patients) for datum in data]).transpose()
    #     rsa_results.update({key: res})

    # ABIDE FIT MODELS (test battery PCA)
    keys_pca = {}
    # keys_pca.update({'updrs(PCA)': ['updrs1_score', 'updrs2_score', 'updrs_totscore']})
    keys_pca.update({'UPDRS (PCA)': [ 'UPDRS I', 'UPDRS II', 'UPDRS Total']})

    # Calculate PCA RSA
    pca_keys = list(keys_pca.keys())
    model_pcas = dict()
    for key in pca_keys:
        arr = np.array(df[keys_pca[key]])
        arr = arr[patients, :]

        idx = ~np.isnan(arr.mean(axis=1))
        mat = arr[idx, :]

        pca = PCA(n_components=1)
        pca_vec = pca.fit_transform(mat)
        rdm = make_RDM(pca_vec)
        model_rdms.update({key: rdm})
        model_idxs.update({key: idx})
        model_pcas.update({key: pca_vec})

        res = np.array([fit_rsa(datum, key,model_idxs,model_rdms,patients) for datum in data]).transpose()
        rsa_results.update({key: res})

        df[key] = 0  # initialize at 0
        for i_rel, i_abs in enumerate(np.nonzero(model_idxs[key])[0]):
            df[key].values[np.nonzero(patients)[0][i_abs]] = model_pcas[key][i_rel]
        return rsa_results
def slice_cmat(data,idx,patients):

    mat = data[patients,:][:,patients] #patients (192,)
    mat = mat[idx,:][:,idx]
    return mat

def fit_rsa(data,key,model_idxs,model_rdms,patients):

    corr = scipy.stats.stats.kendalltau
    r = np.array([corr(get_triu(slice_cmat(data[i,:,:],model_idxs[key],patients)),get_triu(model_rdms[key]))[0] for i in range(10)])
    r = np.arctan(r)
    return r
def slice_cmat_score(data,idx):
    mat = data[idx,:][:,idx]
    return mat

def fit_rsa_score(data,key,model_idxs,model_rdms,patients):
    corr = scipy.stats.stats.kendalltau
    r,p = corr(get_triu(data[key]), get_triu(model_rdms[key]))
    print(r,p)
    r = np.arctan(r)#求反正切值
    return r,p

def fit_model_without_pca(df,model_rdms,model_idxs,cmat_vae_vec_abide, cmat_background_vec_abide, cmat_salient_vec_abide,default_keys):
    patients = df['APPRDX'].values == 1#"PD"
    data = [cmat_vae_vec_abide, cmat_background_vec_abide, cmat_salient_vec_abide]
    rsa_results = dict()
    for key in default_keys:
        print(key)
        res = np.array([fit_rsa(datum, key, model_idxs, model_rdms, patients) for datum in data]).transpose()
        rsa_results.update({key: res})
    return rsa_results

def plot_result(rsa_results,save_path,keys,titles):
    ncols =3 #列
    nrows = int(np.ceil(len(keys) /3)) #行

    plt.figure(figsize=np.array((ncols, nrows)) * 3.5)
    # plt.rc('font', family='Times New Roman')
    for i, key in enumerate(keys):
        ax = plt.subplot(nrows, ncols, i + 1)
        plot_nice_bar(key, rsa_results,
                      ax=ax, figsize=None,
                      dpi=300, fontsize=12,
                      fontsize_star=12,
                      fontweight='bold',
                      line_width=2.5,
                      marker_size=12, title=titles[i])

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.5,
        hspace=0.5) #0.5   .02,/.2

    # plt.suptitle('PPMI RSA RESULTS', fontsize=20, y=.95)
    # plt.show()
    plt.savefig(os.path.join(save_path,"RSA_result.png"),dpi=150)

def compute_T_table(rsa_results,save_path,default_keys,df,num):
    save_path=os.path.join(save_path,'csf_proteins_v2')
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    keys = list(rsa_results.keys())
    columns = ['key', 'VAE>0', 'BG>0', 'SL>0', 'SL>VAE', 'BG>VAE', 'SL>BG', 'BG>SL']
    t_table = pd.DataFrame(np.zeros((len(keys), len(columns))), columns=columns)
    print("t_table",t_table.shape)
    P_list=[]
    T_list=[]
    for i, key in enumerate(default_keys):
        # print(f'{key} / {i}')
        datum = rsa_results[key]
        dg_f = datum.shape[0] - 1
        print(key,dg_f,datum[:, 0].shape)
        t_table['key'][i] = key

        t, p = stats.ttest_1samp(datum[:, 0], 0)
        t_table['VAE>0'][i] = make_tstatement(datum[:, 0], t, p, dg_f)
        # print(make_tstatement(t,p,df))

        t, p = stats.ttest_1samp(datum[:, 1], 0)
        t_table['BG>0'][i] = make_tstatement(datum[:, 1], t, p, dg_f)

        t, p = stats.ttest_1samp(datum[:, 2], 0)
        t_table['SL>0'][i] = make_tstatement(datum[:, 2], t, p, dg_f)

        t, p = stats.ttest_rel(datum[:, 2], datum[:, 0])
        t_table['SL>VAE'][i] = make_tstatement(datum[:, 2] - datum[:, 0], t, p, dg_f, delta=True)

        t, p = stats.ttest_rel(datum[:, 1], datum[:, 0])
        t_table['BG>VAE'][i] = make_tstatement(datum[:, 1] - datum[:, 0], t, p, dg_f, delta=True)
        # print(make_tstatement(t,p,df))

        t, p = stats.ttest_rel(datum[:, 2], datum[:, 1])
        T_list.append(t)
        P_list.append(p)
        t_table['SL>BG'][i] = make_tstatement(datum[:, 2] - datum[:, 1], t, p, dg_f, delta=True)

        t, p = stats.ttest_rel(datum[:, 1], datum[:, 2],)
        t_table['BG>SL'][i] = make_tstatement(datum[:, 1] - datum[:, 2], t, p, dg_f, delta=True)

    cor_p=multipletests(P_list, alpha=0.01, method="fdr_bh")[1]#bonferroni
    t_table['num']=pd.DataFrame(np.array(num))
    t_table['fdr_bh_SL>BG']=pd.DataFrame(cor_p)
    t_table['raw_t'] = pd.DataFrame(T_list)
    t_table['raw_p']=pd.DataFrame(P_list)
    # cvae_p=t_table['SL>BG']
    t_table.to_csv(os.path.join(save_path,'T_tabel_result.csv'),index=False)
    t_table.head(11)
    print(len(t_table))

def make_tstatement(vec, t, p, dg_f, delta=False):
    if p < .001:
        t_statement = f'$\\tau$ = {vec.mean():.2f}, t({dg_f}) = {t:.2f}, p $<$ .001'

    else:
        t_statement = f'$\\tau$ = {vec.mean():.2f}, t({dg_f}) = {t:.2f}, p = {p:.3f}'

    if delta == True:
        t_statement = t_statement.replace('$\\tau$', '$\\Delta\\tau$')
    return t_statement

def computed_siteRSA_vs_score(rsa_result,variable_csv,save_path):
    df = pd.read_csv(variable_csv)
    patients = df['APPRDX'].values == 1
    controls = df['APPRDX'].values == 2
    df.rename(
        columns={'age': 'Age', 'gen': 'Gen', 'moca': 'MoCA', 'updrs1_score': 'UPDRS I', 'rem': 'REM', 'scopa': 'Scopa',
                 'updrs2_score': 'UPDRS II', 'updrs3_score': 'UPDRS III', 'updrs_totscore': 'UPDRS Total',
                 'Serum_NFL': 'Serum_NfL', 'duration': 'Duration',
                 'info-sxdt': 'Info-SXDT', "asyn": 'α-Synuclein', "tau": "T-Tau",
                 "ptau": "P-Tau", "abeta": "A{}42".format(r'$\beta$'), "mean_putamen": "Mean_Putamen",
                 "mean_caudate": "Mean_Caudate", "mean_striatum": "Mean_Striatum"
            , "EDUCYRS": "Education"}, inplace=True)  # 'updrs1':'UPDRS 1','info-pddxdt': 'Info-PDDXDT',
    score_list = ["SITE", "Age", "Gen", "Education", "Mean_Putamen", 'Serum_NfL', "UPDRS III", "MoCA",
    "α-Synuclein", "T-Tau", "P-Tau", "A{}42".format(r'$\beta$')]
    site_rsa=rsa_result['SITE']

    result=pd.DataFrame(np.array(score_list),columns={"score"})
    t_list=[]
    p_list=[]
    for i in range(len(score_list)):
        t, p = stats.pearsonr(site_rsa[:,2], rsa_result[score_list[i]][:,2])
        t_list.append(t)
        p_list.append(p)
    result['T value']=pd.DataFrame(np.array(t_list))
    result['P value']=pd.DataFrame(np.array(p_list))
    print(result.shape)
    result.to_csv(os.path.join(save_path,"site_vs_score_t_p_specific_update_all_pearsons.csv"),index=False)

def computed_RSA_between_score(rsa_result,variable_csv,save_path):
    df = pd.read_csv(variable_csv)
    patients = df['APPRDX'].values == 1
    controls = df['APPRDX'].values == 2
    df.rename(
        columns={'age': 'Age', 'gen': 'Gen', 'moca': 'MoCA', 'updrs1_score': 'UPDRS I', 'rem': 'REM', 'scopa': 'Scopa',
                 'updrs2_score': 'UPDRS II', 'updrs3_score': 'UPDRS III', 'updrs_totscore': 'UPDRS Total',
                 'Serum_NFL': 'Serum_NfL', 'duration': 'Duration',
                 'info-sxdt': 'Info-SXDT', "asyn": 'α-Synuclein', "tau": "T-Tau",
                 "ptau": "P-Tau", "abeta": "A{}42".format(r'$\beta$'), "mean_putamen": "Mean_Putamen",
                 "mean_caudate": "Mean_Caudate", "mean_striatum": "Mean_Striatum"
            , "EDUCYRS": "Education"}, inplace=True)  # 'updrs1':'UPDRS 1','info-pddxdt': 'Info-PDDXDT',
    score_list = ["SITE", "Age", "Gen", "Education", "Mean_Putamen", 'Serum_NfL', "UPDRS III", "MoCA",
    "α-Synuclein", "T-Tau", "P-Tau", "A{}42".format(r'$\beta$')]
    site_rsa=rsa_result['SITE']

    result=pd.DataFrame(np.array(score_list),columns={"score"})
    t_list=[]
    p_list=[]
    for i in range(len(score_list)):
        # inVec = df[score_list[i]].values[patients]
        # idx = ~np.isnan(inVec)
        # inVec = inVec[idx]
        # s, sp = stats.levene(site_rsa[:,1], rsa_result[score_list[i]][:,1] )
        # if sp > 0.05:
        #     t,pvalue = stats.ttest_ind(a=site_rsa[:,1], b=rsa_result[score_list[i]][:,1], equal_var=True)
        # else:
        #     t,pvalue = stats.ttest_ind(a=site_rsa[:,1], b=rsa_result[score_list[i]][:,1], equal_var=False)
        t, p = stats.pearsonr(site_rsa[:,2], rsa_result[score_list[i]][:,2])
        t_list.append(t)
        p_list.append(p)
    result['T value']=pd.DataFrame(np.array(t_list))
    result['P value']=pd.DataFrame(np.array(p_list))
    print(result.shape)
    result.to_csv(os.path.join(save_path,"site_vs_score_t_p_specific_update_all_pearsons.csv"),index=False)

def computed_RSA_site_vs_score(variable_csv,save_path):
    df=pd.read_csv(variable_csv)
    patients = df['APPRDX'].values == 1
    controls = df['APPRDX'].values == 2
    df.rename(
        columns={'age': 'Age', 'gen': 'Gen', 'moca': 'MoCA', 'updrs1_score': 'UPDRS I', 'rem': 'REM', 'scopa': 'Scopa',
                 'updrs2_score': 'UPDRS II', 'updrs3_score': 'UPDRS III', 'updrs_totscore': 'UPDRS Total',
                 'Serum_NFL': 'Serum_NfL', 'duration': 'Duration',
                 'info-sxdt': 'Info-SXDT', "asyn": 'α-Synuclein', "tau": "T-Tau",
                 "ptau": "P-Tau", "abeta": "A{}42".format(r'$\beta$'), "mean_putamen": "Mean_Putamen",
                 "mean_caudate": "Mean_Caudate", "mean_striatum": "Mean_Striatum"
            , "EDUCYRS": "Education"}, inplace=True)  # 'updrs1':'UPDRS 1','info-pddxdt': 'Info-PDDXDT',
    score_list = ["Age", "Gen", "Education", "Mean_Putamen", 'Serum_NfL', "UPDRS III", "MoCA",
                  "α-Synuclein", "T-Tau", "P-Tau", "A{}42".format(r'$\beta$'),'race_raw','info-pddxdt',"Mean_Striatum","Mean_Caudate",]
    scales_ = ['ratio', 'ordinal', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio',
               'ratio',  'ordinal', 'ratio','ratio', 'ratio']
    df=df.iloc[patients]
    print("df.shape",df.shape)
    # site_data=df.iloc[patients]["SITE"].values
    # site_data=np.array(site_data)
    # site_data=np.expand_dims(site_data,axis=1)
    # print(site_data.shape) #(485, 1)
    # site_data=np.array([squareform(pdist(site_data, metric='euclidean'))])
    # print(site_data.shape) #(1, 485, 485)

    # site_inVec = df["SITE"].values[patients]
    # site_idx = ~np.isnan(site_inVec)
    # site_inVec = site_inVec[site_idx] #选出非nan
    # site_rdm = make_RDM(site_inVec, data_scale="ratio")
    # print(site_rdm.shape) #(289, 289)
    model_rdms = dict()
    model_idxs = dict()

    site_rdms=dict()
    # df=df[site_idx]
    for i in range(len(score_list)):
        print(score_list[i])
        # inVec = df[score_list[i]].values[patients]
        # idx = ~np.isnan(inVec)
        # inVec = inVec[idx]
        # inVec = inVec[site_idx]
        score_site_data=df[[score_list[i],"SITE"]]
        print(score_site_data.shape)
        score_site=score_site_data.dropna(subset=[score_list[i], 'SITE'])
        this_rdm = make_RDM(score_site[score_list[i]].values, data_scale=scales_[i])
        site_rdm=make_RDM(score_site['SITE'].values,data_scale="ratio")
        print("this_rdm",this_rdm.shape,site_rdm.shape) #(321, 321)
        model_rdms.update({score_list[i]: this_rdm})
        # model_idxs.update({score_list[i]: idx})
        site_rdms.update({score_list[i]: site_rdm})

    result = pd.DataFrame(np.array(score_list), columns={"score"})
    t_list = []
    p_list = []
    for key in score_list:
        print(key)
        res,p = fit_rsa_score(site_rdms, key, model_idxs, model_rdms, patients)
        # print("res.shape",res.shape)
        t_list.append(res)
        p_list.append(p)
    result['T value'] = pd.DataFrame(np.array(t_list))
    result['P value'] = pd.DataFrame(np.array(p_list))
    print(result.shape)
    print(result)
    result.to_csv(os.path.join(save_path, "site_vs_score_RSA.csv"), index=False)
    return result, df,score_list
    # return df, model_rdms, model_idxs
def computed_RSA_socre_vs_score(variable_csv,save_path):
    df=pd.read_csv(variable_csv)
    patients = df['APPRDX'].values == 1
    controls = df['APPRDX'].values == 2
    # score_list = [ "age", "gen", "rem", "scopa", "moca", "updrs1_score",
    #                 "updrs2_score", "updrs3_score", "updrs_totscore", 'serum_NFL',  "duration","mean_caudate","mean_putamen","mean_striatum","CAUDATE_R","CAUDATE_L","sft",'bjlot','lns',"hvlt_immediaterecall"]
    # score_list =["SITE", "Age", "Gen", "Education", "Mean_Putamen", 'Serum_NfL', "UPDRS III", "MoCA",
    # "α-Synuclein", "T-Tau", "P-Tau", "A{}42".format(r'$\beta$')]
    score_list = ['DATSCAN_PUTAMEN_L', 'DATSCAN_PUTAMEN_R']
    # score_list=[ "DATSCAN_CAUDATE_L",'DATSCAN_CAUDATE_R']
    scales_ = ['ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio', 'ratio',
               'ratio', 'ratio', ]  # supplementary

    df=df.iloc[patients]
    print("df.shape",df.shape)
    # site_data=df.iloc[patients]["SITE"].values
    # site_data=np.array(site_data)
    # site_data=np.expand_dims(site_data,axis=1)
    # print(site_data.shape) #(485, 1)
    # site_data=np.array([squareform(pdist(site_data, metric='euclidean'))])
    # print(site_data.shape) #(1, 485, 485)

    # site_inVec = df["SITE"].values[patients]
    # site_idx = ~np.isnan(site_inVec)
    # site_inVec = site_inVec[site_idx] #选出非nan
    # site_rdm = make_RDM(site_inVec, data_scale="ratio")
    # print(site_rdm.shape) #(289, 289)
    model_rdms = dict()
    model_idxs = dict()

    site_rdms=dict()
    # df=df[site_idx]
    # score_site_data = df[["DATSCAN_PUTAMEN_L", "DATSCAN_PUTAMEN_R"]]
    # score_site = score_site_data.dropna(subset=["DATSCAN_PUTAMEN_L", 'DATSCAN_PUTAMEN_R'])
    # this_rdm = make_RDM(score_site["DATSCAN_PUTAMEN_L"].values, data_scale='ratio')
    # site_rdm = make_RDM(score_site['DATSCAN_PUTAMEN_R'].values, data_scale="ratio")
    # site_rdms.update({'DATSCAN_PUTAMEN_L': site_rdm})
    # model_rdms.update({'DATSCAN_PUTAMEN_L': this_rdm})
    # res, p = fit_rsa_score(site_rdms, 'DATSCAN_PUTAMEN_L', 0, model_rdms, patients)
    # print("putamen",res,p)#0.09542114211853496 0.0

    score_site_data = df[["DATSCAN_CAUDATE_L", "DATSCAN_CAUDATE_R"]]
    score_site = score_site_data.dropna(subset=["DATSCAN_CAUDATE_L", 'DATSCAN_CAUDATE_R'])
    this_rdm = make_RDM(score_site["DATSCAN_CAUDATE_L"].values, data_scale='ratio')
    site_rdm = make_RDM(score_site['DATSCAN_CAUDATE_R'].values, data_scale="ratio")
    site_rdms.update({'DATSCAN_CAUDATE_L': site_rdm})
    model_rdms.update({'DATSCAN_CAUDATE_L': this_rdm})
    res, p = fit_rsa_score(site_rdms, 'DATSCAN_CAUDATE_L', 0, model_rdms, patients)
    print("CAUDATE", res, p)  # 0.2857252808964741 0.0



    for i in range(len(score_list)):
        print(score_list[i])
        # inVec = df[score_list[i]].values[patients]
        # idx = ~np.isnan(inVec)
        # inVec = inVec[idx]
        # inVec = inVec[site_idx]
        score_site_data=df[[score_list[i],"DATSCAN_PUTAMEN_L"]]
        print(score_site_data.shape)
        score_site=score_site_data.dropna(subset=[score_list[i], 'DATSCAN_PUTAMEN_L'])
        this_rdm = make_RDM(score_site[score_list[i]].values, data_scale=scales_[i])
        site_rdm=make_RDM(score_site['DATSCAN_PUTAMEN_L'].values,data_scale="ratio")
        # print("this_rdm",this_rdm.shape,site_rdm.shape) #(321, 321)
        model_rdms.update({score_list[i]: this_rdm})
        # model_idxs.update({score_list[i]: idx})
        site_rdms.update({score_list[i]: site_rdm})

    result = pd.DataFrame(np.array(score_list), columns={"score"})
    t_list = []
    p_list = []
    for key in score_list:
        print(key)
        res,p = fit_rsa_score(site_rdms, key, model_idxs, model_rdms, patients)
        # print("res.shape",res.shape)
        t_list.append(res)
        p_list.append(p)
    # result['T value'] = pd.DataFrame(np.array(t_list))
    # result['P value'] = pd.DataFrame(np.array(p_list))
    # print(result.shape)
    # result.to_csv(os.path.join(save_path, "site_vs_score_RSA.csv"), index=False)
    # return result, df,score_list
    # # return df, model_rdms, model_idxs


if __name__ == '__main__':
    default_keys = [ "SITE", "Age", "Gen",'Serum_NfL',"α-Synuclein","T-Tau","P-Tau","A{}42".format(r'$\beta$'),
                   "UPDRS III","MoCA","Mean_Putamen","Mean_Striatum","Mean_Caudate",'PUTAMEN_R','PUTAMEN_L',
                     'r_striatum','l_striatum',"CAUDATE_R","CAUDATE_L",'Duration','info-pddxdt'] #"Mean_Caudate","Mean_Striatum"

    draw_default_keys=[ "SITE", "Age", "Gen",'Serum_NfL',"α-Synuclein","T-Tau","P-Tau","A{}42".format(r'$\beta$'),
                    "UPDRS III","MoCA","Mean_Putamen","Mean_Striatum","Mean_Caudate",'PUTAMEN_R','PUTAMEN_L',
                        'r_striatum','l_striatum',"CAUDATE_R","CAUDATE_L",'Duration']
    main_default_keys=[ "SITE", "Age", "Gen","Education","Mean_Putamen",'Serum_NfL', "UPDRS III","MoCA",
                        "α-Synuclein","T-Tau","P-Tau","A{}42".format(r'$\beta$')
                  ]
    draw_main_default_keys = ["SITE", "Age", "Gen","Education","Putamen SBR",'Serum_NfL', "UPDRS III","MoCA",
                        "α-Synuclein","T-Tau","P-Tau","A{}42".format(r'$\beta$')]
    weight_list = ['latent_vecs100_cvae_100_1_0.9_0.001.npz', 'latent_vecs100_cvae_150_1_0.9_0.001.npz',
                   'latent_vecs100_cvae_150_1_0.9_0.0001.npz',
                   'latent_vecs100_cvae_100_1_0.9_0.0001.npz', 'latent_vecs100_cvae_300_1_0.9_0.001.npz',
                   'latent_vecs100_cvae_300_1_0.9_0.0001.npz']
    supplement_default_keys=['race_raw','info-pddxdt',"Mean_Striatum","Mean_Caudate"]#,"DATSCAN_CAUDATE_L",'DATSCAN_CAUDATE_R']#''DATSCAN_PUTAMEN_L_ANT','DATSCAN_PUTAMEN_R_ANT',PUTAMEN_L','PUTAMEN_R','l_striatum','r_striatum',"CAUDATE_L","CAUDATE_R",
    supplement_draw_keys=['Race',"Duration",'Striatum SBR',"Caudate SBR"]
    #'Striatum_L SBR','Striatum_R SBR', ,'DATSCAN_PUTAMEN_R_ANT','DATSCAN_PUTAMEN_L_ANT',"DATSCAN_CAUDATE_R",'DATSCAN_CAUDATE_L','DATSCAN_PUTAMEN_R','DATSCAN_PUTAMEN_L'
    save_path='./step3_RSA_739'#_MNI_iqr
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    npz_path='./step2_result_739/latent_vecs10_cvae_bspline_10_1_0.9_0.001.npz'
    variable_csv='../csf_pqtl_protein_data_select_corr_5variables_ols.csv'
    da=pd.read_csv(variable_csv)
    supplement_default_keys_csf=da.columns.tolist()[2:]
    print(supplement_default_keys_csf[:5])
    num_csf=len(supplement_default_keys_csf)
    name=npz_path.split('/')[-1].split('.npz')[0]
    save_path=os.path.join(save_path,name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    cmat_salient_vec_abide, cmat_background_vec_abide, cmat_vae_vec_abide=generate_dissimilarity_metrics(npz_path,save_path)
    df, model_rdms, model_idxs,num=load_legend_make_model(supplement_default_keys_csf,variable_csv,num_csf)
    rsa_results=fit_model_without_pca(df, model_rdms, model_idxs, cmat_vae_vec_abide, cmat_background_vec_abide, cmat_salient_vec_abide,supplement_default_keys_csf)
    plot_result(rsa_results,save_path,supplement_default_keys,supplement_draw_keys)
    print("computed rsa done")
    compute_T_table(rsa_results,save_path,supplement_default_keys_csf,df,num)


    rsa_score,df,draw_default_keys=computed_RSA_site_vs_score(variable_csv,save_path)
    # plot_result(rsa_score, save_path, draw_default_keys, draw_default_keys)
    # compute_T_table(rsa_score,save_path,default_keys,df)

    default_keys_slop=["MoCA",'UPDRS III']
    draw_default_keys_slop=["MoCA", 'UPDRS III']
    variable_csv_slop="../moca_UPDRS_III_slop.csv"
    # #
    cmat_salient_vec_abide, cmat_background_vec_abide, cmat_vae_vec_abide = generate_dissimilarity_metrics(npz_path,
                                                                                                           save_path)
    df, model_rdms, model_idxs,num = load_legend_make_model(default_keys_slop, variable_csv_slop,num_csf=2)
    rsa_results = fit_model_without_pca(df, model_rdms, model_idxs, cmat_vae_vec_abide, cmat_background_vec_abide,
                                        cmat_salient_vec_abide, default_keys_slop)
    plot_result(rsa_results, save_path, draw_default_keys_slop, draw_default_keys_slop)
    compute_T_table(rsa_results, save_path, default_keys_slop, df,2)
    print("done!!")

    computed_siteRSA_vs_score(rsa_results,variable_csv,save_path)
    computed_RSA_between_score(rsa_results,variable_csv,save_path)

