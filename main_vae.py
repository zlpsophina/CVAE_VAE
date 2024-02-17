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
parser = argparse.ArgumentParser(description='vae in ppmi_zheda')
parser.add_argument('--data', default='/home/istbi/HD1/chengyu2_dataset/ppmi_zheda/ppmi_zheda_npy', type=str,
                    help='path to dataset')#metavar='DIR', ADNI2_Slicers_AD_NC
parser.add_argument('--model_name',default='vae_bspline_v2',type=str,choices=('cvae','vae_bspline','vae_bspline_v2'))
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
                    help='beta')
parser.add_argument('--gamma', default=100, type=int, metavar='W',
                    help='game')
parser.add_argument('--disentangle', default=False, type=bool,
                    help='disentangle latent')
parser.add_argument('--latent_size',default=32,type=int,metavar='W',help='latent_size')
parser.add_argument('--print-freq', default=200, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint_dir', default='./checkpoint', type=Path,
                    help='path to checkpoint directory') #metavar='DIR'
parser.add_argument('--final_model_dir', default='./final_model_739', type=Path,
                    metavar='DIR', help='path to final_model directory')
parser.add_argument('--warmup_steps',default=50,type=int,help=" the step of warmup")
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
    device = torch.device("cuda:2")
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
    args.result_dir=os.path.join(args.result_dir,args.model_name+str(args.gamma)+str(args.lr))
    if os.path.exists(args.result_dir) is False:
        os.mkdir(args.result_dir)
    args.result_img_dir=os.path.join(args.result_img_dir,args.model_name+str(args.gamma)+str(args.lr))
    if os.path.exists(args.result_img_dir) is False:
        os.mkdir(args.result_img_dir)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    dir = "tensorboard_739/{}/".format(args.model_name)
    tb_name=str(args.lr)+"_"+str(args.gamma)+"_"+timestamp
    tb_writer = SummaryWriter(log_dir=dir + tb_name)
    model=VAE(input_shape=(64,64,64,1),latent_size=args.latent_size,
                                            batch_size = args.batch_size,
                                            disentangle=args.disentangle,
                                            gamma=args.gamma,
                                            kernel_size = 3,
                                            filters = 48,
                                            intermediate_dim = 128,
                                            nlayers = 2,
                                            bias=True).to(device)

    optimizer = optim.Adam(model.parameters(), eps=1e-07,lr=args.lr, betas=(args.beta, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    checkpiont_path = os.path.join(args.checkpoint_dir,
                                   'checkpoint_{}_{}_{}.pth'.format(args.gamma,args.lr, args.beta))
    if os.path.exists(checkpiont_path):
        ckpt = torch.load(checkpiont_path,map_location='cpu')
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0

    data_list,data_loader=generator_loader_vae(args)
    loss_function = nn.MSELoss()
    mse_list=[]
    loss_list=[]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # data_loader = tqdm(data_loader)
        loss_metric=[]
        for step, image in enumerate(data_loader, start=epoch * len(data_loader)):
            z_mean, z_log_var,z,z_out_shape,decoder_out=model(image.to(device)) #.to(device)
            if args.disentangle:
                loss,loss_metric=model.loss(image.to(device), z_mean, z_log_var,z,decoder_out)
            else:
                loss = model.loss(image.to(device), z_mean.to(device), z_log_var, z, decoder_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # mse = ((data_batch - vae.predict(data_batch)[:, :, :, :, 0]) ** 2).mean()
            loss_mse=loss_function(image[:, 0,:, :, :].to(device),decoder_out[:, 0,:, :, :])
            print("epoch {} step {} mse {} total loss {}".format(epoch, step, loss_mse.item(),loss.item()))
            if loss_mse < 0.005:
                break
            if np.mod(step, 100) == 0:  # Plot training progress
                loss_list.append(loss.item())
                mse_list.append(loss_mse.item())

                im1 = data_list[0]
                im1 = np.load(im1)
                im = im1
                im = np.expand_dims(im, axis=0)
                im = np.expand_dims(im, axis=0) #(1,1,64,64,64)
                im=torch.FloatTensor(im)
                im = model(im.to(device))[-1]
                im=im.cpu().detach().numpy()
                im=im[:,0,:, :,:]
                im=im[0,32,:,:]
                im1=im1[32,:,:]
                pg = optimizer.param_groups
                lr_classifier = pg[0]['lr']
                # plot_trainProgress(loss_list, im, im1,args.result_img_dir,epoch)
                # pickle.dump(loss_list, open(os.path.join(args.result_dir,'{}_{}_loss.pickle'.format(epoch,step)), 'wb'))
                tb_writer.add_scalar("train_loss", loss.item(), epoch)
                tb_writer.add_scalar('lr', lr_classifier, epoch)
                tb_writer.add_scalar('mse_loss',   loss_mse.item(),epoch)
        scheduler.step()

        # state = dict(epoch=epoch + 1, classifier=model.state_dict(), optimizer=optimizer.state_dict(),
        #              scheduler=scheduler.state_dict())
        # torch.save(state, os.path.join(args.checkpoint_dir,
        #                                '{}_{}_{}_{}.pth'.format("checkpoint",args.gamma,args.lr,args.beta)),_use_new_zipfile_serialization=True)

    # tb_writer.add_graph(model, [image.to(device)])
    mse_loos_np=np.array(mse_list)
    loss_list_np=np.array(loss_list)
    np.save(os.path.join(args.result_dir,"{}_mse_loss.npy".format(time_now)),mse_loos_np)
    np.save(os.path.join(args.result_dir, "{}_loss_list.npy".format(time_now)), loss_list_np)
    torch.save(model.state_dict(), os.path.join(args.final_model_dir, '{}_{}_{}_{}.pth'.format(args.model_name,args.gamma,args.beta,args.lr)))

if __name__ == '__main__':
    main()




