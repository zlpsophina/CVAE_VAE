import numpy as np
import pandas as pd
import ants

import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from importlib import reload
import pickle
import helper_funcs;reload(helper_funcs);from helper_funcs import *
import cave_vae_models
reload(cave_vae_models)
from cave_vae_models import *
reload(helper_funcs)
from helper_funcs import *
from tqdm import tqdm

def load_data():
    df=pd.read_csv('../ppmi_IQR_BL_name_id_variable.csv')
    patients = df['APPRDX'].values == 2#'PD'
    df_asd = df.iloc[patients]
    data=np.load('../fast_ori_irq.npy')
    return data,patients,df,df_asd

def load_cvae(weight_path,model_name,gamma_model):
    # LOAD CVAE MODEL
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
def get_recon_and_twin(inMat,model):
    inMat=np.expand_dims(inMat,axis=1)
    inMat=torch.FloatTensor(inMat)
    model.eval()
    z = model.encoder_Z(inMat)[2]
    s = model.encoder_S(inMat)[2]
    zeros = np.zeros(s.shape)
    zeros=torch.FloatTensor(zeros)
    z_s=torch.cat([z,s],dim=1)
    z_zere=torch.cat([z, zeros],dim=1)
    recon = model.decoder(z_s)[:,0,:,:,:]
    twin = model.decoder(z_zere)[:,0,:,:,:]
    recon=recon.detach().numpy()
    twin= twin.detach().numpy()
    print("recon",recon.shape,'twin',twin.shape) #recon (330, 64, 64, 64) twin (330, 64, 64, 64)
    return recon, twin

def plot_brain_image(inMat,recon,twin,save_path):
    s = 0
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(inMat[s, :, :, 40],cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('input')
    plt.subplot(1, 3, 2)
    plt.imshow(recon[s, :, :, 40],cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('reconstruction')
    plt.subplot(1, 3, 3)
    plt.imshow(twin[s, :, :, 40],cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('TD twin')
    plt.savefig(os.path.join(save_path, 'input_recon_TD_twins_brain_z.jpg'))
    plt.close()

    s = 0
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(np.rot90(inMat[s, 45, :, :]),cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('input')
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(recon[s, 45, :, :]),cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('reconstruction')
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(twin[s, 45, :, :]),cmap='gray')
    [ticks([]) for ticks in [plt.xticks, plt.yticks]]
    plt.title('TD twin')
    plt.savefig(os.path.join(save_path,'input_recon_TD_twins_brain_x.jpg'))
    plt.close()


def load_template(save_path):
    template = ants.image_read('../Data/MNI152_T1_2mm_brain.nii.gz')#MNI152_T1_2mm_brain.nii.gz
    template_gm = ants.image_read('../Data/c1Atlas_brain_2mm.nii')
    template_wm = ants.image_read('../Data/c2Atlas_brain_2mm.nii')
    template_gw = template_gm + template_wm

    interp_type =4#1 #4  # bSpline

    template = template.resample_image(resample_params=(64, 64, 64), use_voxels=True, interp_type=interp_type)
    template_gw = template_gw.resample_image(resample_params=(64, 64, 64), use_voxels=True, interp_type=interp_type)

    template[template < .1] = 0
    template_gw[template_gw < .1] = 0
    template.plot_ortho(flat=True, xyz_lines=False, orient_labels=False, title='template')
    # plt.savefig(os.path.join(save_path,'template.jpg'))
    # plt.close()
    # template.plot_ortho(template_gw, flat=True, xyz_lines=False, orient_labels=False, title='template')
    # plt.savefig(os.path.join(save_path,'template_gw.jpg'))
    # plt.close()
    return template,template_gw


def get_Js(invec,df,patients,recon,twin,template,save_path, ofldr='jacobians'):

    # home = os.getenv("HOME")
    home=save_path
    tmp_dir = os.path.join(save_path,'scratch')
    if os.path.exists(tmp_dir) is False:
        os.mkdir(tmp_dir)
    tmp_dir=os.path.join(tmp_dir,'ants_files')
    if os.path.exists(tmp_dir) is False:
        os.mkdir(tmp_dir)

    interpolator = 'bSpline'#"nearestNeighbor" #'bSpline'
    interp_type = 4#1#4

    Js, normed_t1s, normed_recons, normed_twins, nativeJs = [], [], [], [], []
    recon_brains, twin_brains, t1s = [], [], []

    # recon,twin = get_corner_brains(invec)
    res = dict()

    for i in tqdm(range(len(invec))):
        sub = df['file_name'].values[patients][invec[i]]
        t1=ants.image_read('../{}_brain.nii.gz'.format(sub))
        t1 = t1.resample_image(resample_params=(64, 64, 64), use_voxels=True, interp_type=interp_type)
        # t1 = ants.iMath_normalize(t1)

        recon_mat = recon[i, :, :, :]
        twin_mat = twin[i, :, :, :]

        recon_brain = t1.new_image_like(recon_mat)
        twin_brain = t1.new_image_like(twin_mat)

        twin_brain = ants.iMath_normalize(twin_brain)
        recon_brain = ants.iMath_normalize(recon_brain)

        # rigid match T1, Recon and Twin

        # twin_brain = ants.registration(fixed=t1,moving=twin_brain,type_of_transform='Rigid')['warpedmovout']
        # recon_brain = ants.registration(fixed=t1,moving=recon_brain,type_of_transform='Rigid')['warpedmovout']

        # Match twin to recon
        # ants.registration()
        twin_brain = ants.registration(fixed=recon_brain, moving=twin_brain, type_of_transform='Rigid')['warpedmovout']

        tx2t1 = ants.registration(fixed=t1, moving=recon_brain, type_of_transform='Rigid', outprefix=tmp_dir)
        # Match twin and recon to T1

        recon_brain = ants.apply_transforms(fixed=t1, moving=recon_brain, transformlist=tx2t1['fwdtransforms'],
                                            interpolator=interpolator)

        twin_brain = ants.apply_transforms(fixed=t1, moving=twin_brain, transformlist=tx2t1['fwdtransforms'],
                                           interpolator=interpolator)

        # calculate jacobian in native space

        tx = ants.registration(fixed=recon_brain, moving=twin_brain, type_of_transform='SyN', outprefix=tmp_dir)

        J = ants.create_jacobian_determinant_image(domain_image=recon_brain, tx=tx['fwdtransforms'][0])
        # ants.image_write(J, "./result/jac.nii.gz")
        J = J - 1

        norm = ants.registration(fixed=template, moving=t1, type_of_transform='SyN', outprefix=tmp_dir)

        normed_t1 = ants.apply_transforms(fixed=template, moving=t1, transformlist=norm['fwdtransforms'],
                                          interpolator=interpolator)

        normed_recon = ants.apply_transforms(fixed=template, moving=recon_brain, transformlist=norm['fwdtransforms'],
                                             interpolator=interpolator)

        normed_twin = ants.apply_transforms(fixed=template, moving=twin_brain, transformlist=norm['fwdtransforms'],
                                            interpolator=interpolator)

        normed_J = ants.apply_transforms(fixed=template, moving=J, transformlist=norm['fwdtransforms'],
                                         interpolator=interpolator)


        #SAVE THE RESULTS
        # Where to save everything
        ofdir = os.path.join(home, ofldr)
        print("ofdir",ofdir)

        # MAKE OFDIR IF NOT EXIST
        _ = os.mkdir(ofdir) if not os.path.exists(ofdir) else 0

        # Make a dict
        res = dict()

        res['native_Js'] = J
        res['twin_brains'] = twin_brain
        res['recon_brains'] = recon_brain
        res['t1s'] = t1

        res['normed_Js'] = normed_J
        res['normed_t1s'] = normed_t1
        res['normed_recons'] = normed_recon
        res['normed_twins'] = normed_twin

        for key in list(res.keys()):
            # make a subdir if needed
            _ = os.mkdir(os.path.join(ofdir, key)) if os.path.exists(
                os.path.join(ofdir, key)) == False else 0  # One liner if statement
            res[key].to_filename(os.path.join(ofdir, key, f'{sub}_{key}.nii'))

    return res

def metric2(df_asd,save_path):

    home =save_path# os.getenv("HOME")
    jac_dir = os.path.join(home, 'jacobians')
    folders = os.listdir(jac_dir)

    asd_subs = df_asd['file_name'].values
    res = dict()
    res_keys=[]
    for folder in tqdm(folders):
        res[folder] = normed_Js = [ants.image_read(os.path.join(jac_dir, folder, f'{sub}_{folder}.nii')) for sub in
                                   asd_subs]
        res_keys = list(res.keys())
        print("res_keys",res_keys)
    for key in ['t1s', 'recon_brains', 'twin_brains', 'normed_Js'] :#,  'native_Js','normed_recons', 'normed_t1s','normed_twins']:
        res[key][0].new_image_like(np.array([arr.numpy() for arr in res[key]]).mean(axis=0)).plot_ortho(
            title=f'mean {key}', flat=True, orient_labels=False, xyz_lines=False,filename=os.path.join(save_path, 'mean {}.jpg'.format(key)))
    # cmat_native_Js = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['native_Js']]))
    cmat_twin_brains = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['twin_brains']]))
    cmat_t1s = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['t1s']]))
    cmat_recon_brains = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['recon_brains']]))

    cmat_normed_Js = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['normed_Js']]))

    plt.subplots(2, 4, figsize=(12, 5))
    mats = [ cmat_twin_brains, cmat_t1s, cmat_recon_brains,
            cmat_normed_Js]#cmat_native_Js,, cmat_normed_t1s, cmat_normed_recons, cmat_normed_twins]

    mat_ttls = [ 'cmat_twin_brains', 'cmat_t1s', 'cmat_recon_brains', 'cmat_normed_Js']
                 #'cmat_native_Js',,'cmat_normed_t1s', 'cmat_normed_recons', 'cmat_normed_twins']

    for idx, mat in enumerate(mats):
        plt.subplot(2, 4, idx + 1)
        plt.imshow(mat)
        [tics([]) for tics in [plt.xticks, plt.yticks]]
        plt.title(mat_ttls[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'show_coeff.jpg'))
    plt.close()

    keys = list(res.keys())
    key = keys[0]
    stat = dict()
    for key in keys:
        m = np.array([arr.numpy() for arr in res[key]]).mean(axis=0).flatten()
        sim2mean = np.array([np.corrcoef(arr.numpy().flatten(), m)[0, 1] for arr in res[key]])
        stat['sim2mean' + key] = sim2mean
    print("stat.keys",list(stat.keys()))
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    keys = ['sim2meant1s',
            'sim2meantwin_brains',
            'sim2meanrecon_brains',
            'sim2meannormed_Js',
            ]
            # 'sim2meannative_Js',
    # 'sim2meannormed_recons',
            # 'sim2meannormed_t1s',
            # 'sim2meannormed_twins']

    for i, key in enumerate(keys):
        plt.subplot(2, 4, i + 1)
        plt.hist(stat[key])
        plt.title(key)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'show_similarity.jpg'))
    plt.close()

    print("res_keys",res_keys)
    nsubs = len(res['recon_brains'])
    rec_twim_sim = np.array(
        [np.corrcoef(res['recon_brains'][s].numpy().flatten(), res['twin_brains'][s].numpy().flatten())[0, 1] for s in
         range(nsubs)])
    plt.hist(rec_twim_sim)
    plt.savefig(os.path.join(save_path,'show_rec_twim_similarity.jpg'))
    plt.close()
    print(
        f'mean={rec_twim_sim.mean():.2f},sd={rec_twim_sim.std():.2f},'
        f'min={rec_twim_sim.min():.2f},max={rec_twim_sim.max():.2f}')

def metric(df_asd,save_path):
    home =save_path# os.getenv("HOME")
    jac_dir = os.path.join(home, 'jacobians')
    folders = os.listdir(jac_dir)

    asd_subs = df_asd['file_name'].values

    res = dict()
    res_keys=[]
    for folder in tqdm(folders):
        res[folder] = normed_Js = [ants.image_read(os.path.join(jac_dir, folder, f'{sub.split(".npy")[0]}_{folder}.nii')) for sub in
                                   asd_subs]
        res_keys = list(res.keys())
        print("res_keys",res_keys)
    for key in [ 't1s','recon_brains', 'twin_brains','native_Js', 'normed_t1s', 'normed_recons', 'normed_twins','normed_Js'] :#, ]:
        res[key][0].new_image_like(np.array([arr.numpy() for arr in res[key]]).mean(axis=0)).plot_ortho(
            title=f'mean {key}', flat=True, orient_labels=False, xyz_lines=False,filename=os.path.join(save_path, 'mean {}.jpg'.format(key)))
    cmat_native_Js = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['native_Js']]))
    cmat_twin_brains = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['twin_brains']]))
    cmat_t1s = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['t1s']]))
    cmat_recon_brains = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['recon_brains']]))

    cmat_normed_Js = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['normed_Js']]))
    cmat_normed_t1s = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['normed_t1s']]))
    cmat_normed_recons = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['normed_recons']]))
    cmat_normed_twins = np.corrcoef(np.array([arr.numpy().flatten() for arr in res['normed_twins']]))
    plt.subplots(2, 4, figsize=(12, 5))
    mats = [ cmat_t1s, cmat_recon_brains,cmat_twin_brains, cmat_native_Js,cmat_normed_t1s,
            cmat_normed_Js, cmat_normed_recons, cmat_normed_twins]#]

    mat_ttls = ['cmat_t1s', 'cmat_recon_brains', 'cmat_twin_brains', 'cmat_native_Js', 'cmat_normed_t1s',
                'cmat_normed_Js','cmat_normed_recons', 'cmat_normed_twins',] #

    for idx, mat in enumerate(mats):
        plt.subplot(2, 4, idx + 1)
        plt.imshow(mat)
        [tics([]) for tics in [plt.xticks, plt.yticks]]
        plt.title(mat_ttls[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'show_coeff.jpg'))
    plt.close()

    keys = list(res.keys())
    key = keys[0]
    stat = dict()
    for key in keys:
        m = np.array([arr.numpy() for arr in res[key]]).mean(axis=0).flatten()
        sim2mean = np.array([np.corrcoef(arr.numpy().flatten(), m)[0, 1] for arr in res[key]])
        stat['sim2mean' + key] = sim2mean
    print("stat.keys",list(stat.keys()))
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    keys = [
            'sim2meant1s','sim2meanrecon_brains', 'sim2meantwin_brains',
            'sim2meannative_Js',
            'sim2meannormed_t1s',
            'sim2meannormed_Js',
            'sim2meannormed_recons',
            'sim2meannormed_twins'

            ]
            #

    # ]

    for i, key in enumerate(keys):
        plt.subplot(2, 4, i + 1)
        plt.hist(stat[key])
        plt.title(key)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'show_similarity.jpg'))
    plt.close()

    print("res_keys",res_keys)
    nsubs = len(res["normed_recons"])
    rec_twim_sim = np.array(
        [np.corrcoef(res['normed_recons'][s].numpy().flatten(), res['normed_twins'][s].numpy().flatten())[0, 1] for s in
         range(nsubs)])
    plt.hist(rec_twim_sim)
    plt.savefig(os.path.join(save_path,'show_normed_rec_twim_similarity.jpg'))
    plt.close()
    print(
        f'mean={rec_twim_sim.mean():.2f},sd={rec_twim_sim.std():.2f},'
        f'min={rec_twim_sim.min():.2f},max={rec_twim_sim.max():.2f}')


if __name__ == '__main__':

    weight_path='../final_model_739/cvae_bspline/cvae_bspline_10_1_0.9_0.001.pth'
    loss_path='../result/cvae/12_240_loss.pickle'
    save_path='./step5_make_Jacobian_result_739_nc_161'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    cvae_choice = ['cave']
    name=weight_path.split('/')[-1].split('.pth')[0]
    save_path=os.path.join(save_path,name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    data,patients,df,df_asd=load_data()
    model=load_cvae(weight_path,cvae_choice[0],gamma_model=10)
    inMat = data[patients, :, :, :]
    # print(inMat.shape)
    recon, twin=get_recon_and_twin(inMat,model)
    plot_brain_image(inMat, recon, twin, save_path)
    template, template_gw=load_template(save_path)
    res = get_Js(np.arange(patients.sum()),df,patients,recon,twin,template,save_path)
    metric(df_asd, save_path)
    print("done!!")





