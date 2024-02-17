from pathlib import Path
import argparse
import json
import os
import random
import signal
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import time
import urllib
import numpy as np
from datetime import datetime
from torch import nn, optim
from torchvision import models, transforms
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from cave_vae_models import *
from functions import *
from helper_funcs import *
import tqdm
parser = argparse.ArgumentParser(description='cvae in ppmi_zheda')
parser.add_argument('--data', default='../ppmi_zheda/ppmi_zheda_npy', type=str,
                    help='path to dataset')#metavar='DIR', ADNI2_Slicers_AD_NC
parser.add_argument('--model_name',default='cvae',type=str,choices=('vae','cvae',))
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--beta', default=0.9, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--gamma', default=20, type=int, metavar='W',
                    help='game')
parser.add_argument('--beta_cave',default=1,type=int,metavar='W', help='beta_cave')
parser.add_argument('--disentangle', default=True, type=bool,
                    help='disentangle latent or not')
parser.add_argument('--latent_size',default=16,type=int,metavar='W',help='latent_size')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint_dir', default='./checkpoint', type=Path,
                    help='path to checkpoint directory') #metavar='DIR'
parser.add_argument('--final_model_dir', default='./final_model', type=Path,
                    metavar='DIR', help='path to final_model directory')
parser.add_argument("--result_dir",default="./result",type=Path,help="the way saved result roc and matrix")
parser.add_argument("--result_img_dir",default="./result_img",type=Path,help="the way saved result roc and matrix")
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
def main():
    #single gpu
    start_time = time.time()
    device = torch.device("cuda:3")
    setup_seed(2)
    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
    time_now = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.final_model_dir.mkdir(parents=True,exist_ok=True)
    args.result_dir.mkdir(parents=True,exist_ok=True)
    args.result_img_dir.mkdir(parents=True,exist_ok=True)
    args.checkpoint_dir=os.path.join(args.checkpoint_dir,args.model_name)
    if os.path.exists(args.checkpoint_dir) is False:
        os.mkdir(args.checkpoint_dir)
    args.final_model_dir=os.path.join(args.final_model_dir,args.model_name)
    if os.path.exists(args.final_model_dir) is False:
        os.mkdir(args.final_model_dir)
    args.result_dir=os.path.join(args.result_dir,args.model_name+str(args.gamma)+str(args.beta_cave)+str(args.lr))
    if os.path.exists(args.result_dir) is False:
        os.mkdir(args.result_dir)
    args.result_img_dir=os.path.join(args.result_img_dir,args.model_name+str(args.gamma)+str(args.beta_cave)+str(args.lr))
    if os.path.exists(args.result_img_dir) is False:
        os.mkdir(args.result_img_dir)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    dir = "tensorboard/{}/".format(args.model_name)
    tb_name=str(args.lr)+"_"+str(args.gamma)+"_"+str(args.beta_cave)+"_"+timestamp
    tb_writer = SummaryWriter(log_dir=dir + tb_name)
    model=CVAE(latent_size=args.latent_size,batch_size=args.batch_size,beta=args.beta_cave,gamma=args.gamma,disentangle=args.disentangle).to(device)

    optimizer = optim.Adam(model.parameters(), eps=1e-07,lr=args.lr, betas=(args.beta, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    checkpiont_path = os.path.join(args.checkpoint_dir,
                                   'checkpoint_{}_{}_{}_{}.pth'.format(args.gamma,args.beta_cave,args.lr,args.beta))
    if os.path.exists(checkpiont_path):
        ckpt = torch.load(checkpiont_path,map_location='cpu')
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0

    pd_subs_list,nc_subs_list,data_loader=generator_loader(args)
    loss_function = nn.MSELoss()
    mse_list=[]
    loss_list=[]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # data_loader = tqdm(data_loader)
        for step, ( DX_subs,TD_subs) in enumerate(data_loader, start=epoch * len(data_loader)):
            PD_decoder_out, NC_decoder_out, PD_z_mean, PD_z_log_var,PD_z,\
            PD_s_mean, PD_s_log_var,PD_s, NC_z_mean, NC_z_log_var,NC_z=model(DX_subs.to(device),TD_subs.to(device)) #.to(device)
            loss=model.loss(DX_subs.to(device),TD_subs.to(device),PD_decoder_out,NC_decoder_out,\
                            PD_z,PD_z_mean,PD_z_log_var,PD_s,PD_s_mean,PD_s_log_var,NC_z_mean,NC_z_log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            a=torch.stack((DX_subs.to(device),TD_subs.to(device)),dim=0)[:,:,0,:,:,:]
            b=torch.stack((PD_decoder_out, NC_decoder_out),dim=0)[:,:,0,:,:,:]
            loss_mse=loss_function(a,b)
            print("epoch {} step {} mse {} total loss {}".format(epoch, step, loss_mse.item(),loss.item()))
            if loss_mse < 0.005:
                break
            # im, im1, ss = cvae_query_2(pd_subs_list, model,device)
            loss_list.append(loss.item())
            mse_list.append(loss_mse.item())
            if step % args.print_freq == 0:  # Plot training progress
                # plot_trainProgress(loss_list, im, im1,args.result_img_dir,epoch)
                pg = optimizer.param_groups
                lr_classifier = pg[0]['lr']
                tb_writer.add_scalar('lr',lr_classifier,epoch)
                tb_writer.add_scalar("train_loss", loss.item(), epoch)
                tb_writer.add_scalar('mse_loss',   loss_mse.item(),epoch)

        scheduler.step()
    mse_loos_np=np.array(mse_list)
    loss_list_np=np.array(loss_list)
    np.save(os.path.join(args.result_dir,"{}_mse_loss.npy".format(time_now)),mse_loos_np)
    np.save(os.path.join(args.result_dir, "{}_loss_list.npy".format(time_now)), loss_list_np)
    torch.save(model.state_dict(), os.path.join(args.final_model_dir, '{}_{}_{}_{}_{}.pth'.format(args.model_name,args.gamma,args.beta_cave,args.beta,args.lr)))

if __name__ == '__main__':
    main()





