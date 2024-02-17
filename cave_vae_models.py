import os
import torch
import torch.nn as nn
import pandas
import numpy as np
# from torchsummary import summary
# from tensorboardX import SummaryWriter
# from torchviz import make_dot
# import graphviz

class CVAE(nn.Module):
    def __init__(self, input_shape=(64,64,64,1), latent_size=16, batch_size=64,beta=1,gamma=100, disentangle=True, bias=True):
        super(CVAE, self).__init__()
        self.input_shape=input_shape
        self.image_size, _, _, self.channels = input_shape
        self.kernel_size = 3
        self.intermediate_dim = 128
        self.batch_size=batch_size
        self.latent_size = latent_size
        self.disentangle=disentangle
        self.beta=beta
        self.gamma=gamma
        self.m=int(batch_size/2)
        self.bias=bias


        self.encoder_forward_Z = nn.Sequential(
            nn.Conv3d(self.channels,64,kernel_size=self.kernel_size,stride=2,padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128,kernel_size=self.kernel_size,stride=2,padding=(1,1,1),bias=self.bias),
            nn.ReLU(inplace=True)
        )
        self.encoder_forward_S = nn.Sequential(
            nn.Conv3d(self.channels, 64, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Sequential(nn.Linear(524288, self.intermediate_dim,bias=self.bias),
                                    nn.ReLU(inplace=True))
        #self.ASD_out_shape[1]*self.ASD_out_shape[2]*self.ASD_out_shape[3]*self.ASD_out_shape[4]
        self.mean_linear=nn.Linear(self.intermediate_dim,self.latent_size,bias=self.bias)
        self.log_var_linear=nn.Linear(self.intermediate_dim,self.latent_size,bias=self.bias)

        self.decoder_forward_linear= nn.Sequential(
            nn.Linear(self.latent_size*2,self.intermediate_dim,bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Linear(self.intermediate_dim,self.intermediate_dim*self.latent_size*self.latent_size*self.latent_size,bias=self.bias),
            nn.ReLU(inplace=True)
        )
        ##nn.Linear(self.intermediate_dim, self.ASD_out_shape[1]*self.ASD_out_shape[2]*self.ASD_out_shape[3]*self.ASD_out_shape[4])
        self.decoder_forward_up=nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=self.kernel_size+1, stride=2, padding=(1, 1,1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=self.kernel_size+1, stride=2, padding=(1, 1,1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=self.kernel_size, stride=1, padding=(1, 1,1),bias=self.bias),
            nn.Sigmoid()
        )
        self.discriminator=nn.Sequential(nn.Linear(32,1),
                                         nn.Sigmoid())
        self.loss_func=torch.nn.MSELoss()
    def encoder_Z(self, X):
        # print("X",X.shape)#torch.Size([64, 1, 64, 64, 64])
        z_out = self.encoder_forward_Z(X)
        # print("z_out",z_out.shape) # torch.Size([64, 128, 16, 16, 16])
        z_out_shape=z_out.shape
        self.z_out_shape = z_out.shape
        z_out=torch.flatten(z_out,start_dim=1,end_dim=-1)#z_out.view(z_out.size(0), -1)
        # print("z_out",z_out.shape) # torch.Size([64, 524288])
        z_out=self.linear(z_out)
        z_mean =self.mean_linear(z_out) #torch.Size([64, 16])
        z_log_var = self.log_var_linear(z_out)#torch.Size([64, 16])
        z_z = self.reparameterization(z_mean, z_log_var) #ttorch.Size([64, 16])
        return z_mean, z_log_var,z_z,z_out_shape
    def encoder_S(self, X):
        s_out = self.encoder_forward_S(X)
        s_out_shape=s_out.shape
        s_out=torch.flatten(s_out,start_dim=1,end_dim=-1)
        s_out=self.linear(s_out)
        s_mean =self.mean_linear(s_out)
        s_log_var = self.log_var_linear(s_out)
        s_s = self.reparameterization(s_mean, s_log_var)
        return s_mean, s_log_var,s_s,s_out_shape

    def decoder(self, z):
        z = self.decoder_forward_linear(z)
        b=z.shape[0]
        z=z.reshape(b,128,16,16,16)
        z=self.decoder_forward_up(z)
        return z

    def reparameterization(self, mean, log_var):
        # 从均值和方差分布中进行采样生成Z
        epsilon = torch.randn_like(log_var)
        z = mean + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self,ASD_input,NC_input,ASD_decoder_out,NC_decoder_out,
             ASD_z,ASD_z_mean,ASD_z_log_var,ASD_s,ASD_s_mean,ASD_s_log_var,NC_z_mean,NC_z_log_var):
        # print("ASD_z_mean",ASD_z_mean.shape) # torch.Size([64, 16])
        # print("ASD_z_log_var",ASD_z_log_var.shape)#torch.Size([64, 16])

        if self.disentangle:
            q_score,q_bar_score=self.forward_disentangle(ASD_z,ASD_s)
            tc_loss = torch.log(q_score / (1 - q_score))
            discriminator_loss = - torch.log(q_score) - torch.log(1 - q_bar_score)
        else:
            tc_loss=0
            discriminator_loss=0
        # reconstruction_loss1 = torch.mean(torch.square(X - mu_prime).sum(dim=1))
        reconstruction_loss = self.loss_func(torch.flatten(ASD_input,start_dim=1,end_dim=-1),torch.flatten(ASD_decoder_out,start_dim=1,end_dim=-1))
        reconstruction_loss += self.loss_func(torch.flatten(NC_input,start_dim=1,end_dim=-1), torch.flatten(NC_decoder_out,start_dim=1,end_dim=-1))
        reconstruction_loss *= self.input_shape[0] * self.input_shape[1]# * self.input_shape[2] * self.input_shape[3]


        # reconstruction_loss1=torch.mean(torch.square(ASD_input-ASD_decoder_out).sum(dim=1))#
        # reconstruction_loss2=torch.mean(torch.square(NC_input-NC_decoder_out).sum(dim=1))#
        # reconstruction_loss=reconstruction_loss1+reconstruction_loss2
        #
        # reconstruction_loss1 =self.loss_func(ASD_input,ASD_decoder_out)
        # reconstruction_loss2=self.loss_func(NC_input,NC_decoder_out)
        # reconstruction_loss = reconstruction_loss1 + reconstruction_loss2


        kl_loss = 1 + ASD_z_log_var - torch.square(ASD_z_mean) - torch.exp(ASD_z_log_var)
        kl_loss += 1 + ASD_s_log_var - torch.square(ASD_s_mean) - torch.exp(ASD_s_log_var)
        kl_loss += 1 + NC_z_log_var- torch.square(NC_z_mean) - torch.exp(NC_z_log_var)
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        # print(reconstruction_loss)
        # print(kl_loss)
        # print(tc_loss)
        # print(discriminator_loss)

        cvae_loss = torch.mean(reconstruction_loss + self.beta * kl_loss + self.gamma * tc_loss + discriminator_loss)
        return cvae_loss

    def forward_disentangle(self,ASD_z,ASD_s):
        # print(ASD_ASD_z.shape) #torch.Size([64, 16])
        # print(ASD_NC_z.shape)#torch.Size([64, 16])
        z1 = ASD_z[:self.m, :]#torch.Size([32, 16])
        z2 = ASD_z[self.m:, :]#torch.Size([32, 16])
        s1 = ASD_s[:self.m, :]#torch.Size([32, 16])
        s2 = ASD_s[self.m:, :]#torch.Size([32, 16])

        q_bar1=torch.cat([s1, z2], dim=1)
        q_bar2=torch.cat([s2, z1], dim=1)
        q_bar=torch.cat([q_bar1,q_bar2],dim=0)
        q_1=torch.cat([s1, z1], dim=1)
        q_2=torch.cat([s2, z2], dim=1)
        q=torch.cat([q_1,q_2],dim=0)
        q_bar_score = (self.discriminator(q_bar) + .1) * .85  # +.1 * .85 so that it's 0<x<1
        q_score = (self.discriminator(q) + .1) * .85
        return q_score,q_bar_score

    def forward(self,ASD_input,NC_input):
        # print("ASD_input",ASD_input.shape)#torch.Size([64, 1, 64, 64, 64])
        # print("NC_input",NC_input.shape)#torch.Size([64, 1, 64, 64, 64])
        ASD_z_mean, ASD_z_log_var, ASD_z, ASD_out_shape = self.encoder_Z(ASD_input)
        ASD_s_mean, ASD_s_log_var, ASD_s, _ = self.encoder_S(ASD_input)
        NC_z_mean, NC_z_log_var, NC_z, _ = self.encoder_Z(NC_input)
        z = torch.cat([ASD_z, ASD_s], dim=1)
        ASD_decoder_out = self.decoder(z)
        zeros = torch.zeros_like(ASD_z)
        z_nc=torch.cat([NC_z, zeros],dim=1)
        NC_decoder_out = self.decoder(z_nc)
        return ASD_decoder_out,NC_decoder_out,ASD_z_mean, ASD_z_log_var, ASD_z,ASD_s_mean, ASD_s_log_var, ASD_s,NC_z_mean, NC_z_log_var, NC_z

    class CVAE_v2(nn.Module):

        def __init__(self, input_shape=(64, 64, 64, 1), latent_size=16, batch_size=64, beta=1, gamma=100,
                     disentangle=True, bias=True):
            super(CVAE_v2, self).__init__()
            self.input_shape = input_shape
            self.image_size, _, _, self.channels = input_shape
            self.kernel_size = 3
            self.intermediate_dim = 128
            self.batch_size = batch_size
            self.latent_size = latent_size
            self.disentangle = disentangle
            self.beta = beta
            self.gamma = gamma
            self.m = int(batch_size / 2)
            self.bias = bias

            self.encoder_forward_Z = nn.Sequential(
                nn.Conv3d(self.channels, 64, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1), bias=self.bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1), bias=self.bias),
                nn.ReLU(inplace=True)
            )
            self.encoder_forward_S = nn.Sequential(
                nn.Conv3d(self.channels, 64, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1), bias=self.bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, kernel_size=self.kernel_size, stride=2, padding=(1, 1, 1), bias=self.bias),
                nn.ReLU(inplace=True)
            )
            self.linear = nn.Sequential(nn.Linear(524288, self.intermediate_dim, bias=self.bias),
                                        nn.ReLU(inplace=True))
            # self.ASD_out_shape[1]*self.ASD_out_shape[2]*self.ASD_out_shape[3]*self.ASD_out_shape[4]
            self.mean_linear = nn.Linear(self.intermediate_dim, self.latent_size, bias=self.bias)
            self.log_var_linear = nn.Linear(self.intermediate_dim, self.latent_size, bias=self.bias)

            self.decoder_forward_linear = nn.Sequential(
                nn.Linear(self.latent_size * 2, self.intermediate_dim, bias=self.bias),
                nn.ReLU(inplace=True),
                nn.Linear(self.intermediate_dim,
                          self.intermediate_dim * self.latent_size * self.latent_size * self.latent_size,
                          bias=self.bias),
                nn.ReLU(inplace=True)
            )
            ##nn.Linear(self.intermediate_dim, self.ASD_out_shape[1]*self.ASD_out_shape[2]*self.ASD_out_shape[3]*self.ASD_out_shape[4])
            self.decoder_forward_up = nn.Sequential(
                nn.ConvTranspose3d(128, 32, kernel_size=self.kernel_size + 1, stride=2, padding=(1, 1, 1),
                                   bias=self.bias),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(32, 16, kernel_size=self.kernel_size + 1, stride=2, padding=(1, 1, 1),
                                   bias=self.bias),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(16, 1, kernel_size=self.kernel_size, stride=1, padding=(1, 1, 1), bias=self.bias),
                nn.Sigmoid()
            )
            self.discriminator = nn.Sequential(nn.Linear(32, 1),
                                               nn.Sigmoid())

            self.loss_func = torch.nn.MSELoss()

        def encoder_Z(self, X):
            # print("X",X.shape)#torch.Size([64, 1, 64, 64, 64])
            z_out = self.encoder_forward_Z(X)
            # print("z_out",z_out.shape) # torch.Size([64, 128, 16, 16, 16])
            z_out_shape = z_out.shape
            self.z_out_shape = z_out.shape
            z_out = torch.flatten(z_out, start_dim=1, end_dim=-1)  # z_out.view(z_out.size(0), -1)
            # print("z_out",z_out.shape) # torch.Size([64, 524288])
            z_out = self.linear(z_out)
            z_mean = self.mean_linear(z_out)  # torch.Size([64, 16])
            z_log_var = self.log_var_linear(z_out)  # torch.Size([64, 16])
            z_z = self.reparameterization(z_mean, z_log_var)  # ttorch.Size([64, 16])
            return z_mean, z_log_var, z_z, z_out_shape

        def encoder_S(self, X):
            s_out = self.encoder_forward_S(X)
            s_out_shape = s_out.shape
            s_out = torch.flatten(s_out, start_dim=1, end_dim=-1)
            s_out = self.linear(s_out)
            s_mean = self.mean_linear(s_out)
            s_log_var = self.log_var_linear(s_out)
            s_s = self.reparameterization(s_mean, s_log_var)
            return s_mean, s_log_var, s_s, s_out_shape

        def decoder(self, z):
            z = self.decoder_forward_linear(z)
            b = z.shape[0]
            z = z.reshape(b, 128, 16, 16, 16)
            z = self.decoder_forward_up(z)
            return z

        def reparameterization(self, mean, log_var):
            # 从均值和方差分布中进行采样生成Z
            epsilon = torch.randn_like(log_var)
            z = mean + epsilon * torch.sqrt(log_var.exp())
            return z

        def loss(self, ASD_input, NC_input, ASD_decoder_out, NC_decoder_out,
                 ASD_z, ASD_z_mean, ASD_z_log_var, ASD_s, ASD_s_mean, ASD_s_log_var, NC_z_mean, NC_z_log_var):
            # print("ASD_z_mean",ASD_z_mean.shape) # torch.Size([64, 16])
            # print("ASD_z_log_var",ASD_z_log_var.shape)#torch.Size([64, 16])

            if self.disentangle:
                q_score, q_bar_score = self.forward_disentangle(ASD_z, ASD_s)
                tc_loss = torch.log(q_score / (1 - q_score))
                discriminator_loss = - torch.log(q_score) - torch.log(1 - q_bar_score)
            else:
                tc_loss = 0
                discriminator_loss = 0
            # reconstruction_loss1 = torch.mean(torch.square(X - mu_prime).sum(dim=1))
            reconstruction_loss = self.loss_func(torch.flatten(ASD_input, start_dim=1, end_dim=-1),
                                                 torch.flatten(ASD_decoder_out, start_dim=1, end_dim=-1))
            reconstruction_loss += self.loss_func(torch.flatten(NC_input, start_dim=1, end_dim=-1),
                                                  torch.flatten(NC_decoder_out, start_dim=1, end_dim=-1))
            reconstruction_loss *= self.input_shape[0] * self.input_shape[
                1]  # * self.input_shape[2] * self.input_shape[3]

            # reconstruction_loss1=torch.mean(torch.square(ASD_input-ASD_decoder_out).sum(dim=1))#
            # reconstruction_loss2=torch.mean(torch.square(NC_input-NC_decoder_out).sum(dim=1))#
            # reconstruction_loss=reconstruction_loss1+reconstruction_loss2
            #
            # reconstruction_loss1 =self.loss_func(ASD_input,ASD_decoder_out)
            # reconstruction_loss2=self.loss_func(NC_input,NC_decoder_out)
            # reconstruction_loss = reconstruction_loss1 + reconstruction_loss2

            kl_loss = 1 + ASD_z_log_var - torch.square(ASD_z_mean) - torch.exp(ASD_z_log_var)
            kl_loss += 1 + ASD_s_log_var - torch.square(ASD_s_mean) - torch.exp(ASD_s_log_var)
            kl_loss += 1 + NC_z_log_var - torch.square(NC_z_mean) - torch.exp(NC_z_log_var)
            kl_loss = torch.sum(kl_loss, dim=1)
            kl_loss *= -0.5
            # print(reconstruction_loss)
            # print(kl_loss)
            # print(tc_loss)
            # print(discriminator_loss)

            cvae_loss = torch.mean(
                reconstruction_loss + self.beta * kl_loss + self.gamma * tc_loss + discriminator_loss)
            return cvae_loss

        def forward_disentangle(self, ASD_z, ASD_s):
            # print(ASD_ASD_z.shape) #torch.Size([64, 16])
            # print(ASD_NC_z.shape)#torch.Size([64, 16])
            z1 = ASD_z[:self.m, :]  # torch.Size([32, 16])
            z2 = ASD_z[self.m:, :]  # torch.Size([32, 16])
            s1 = ASD_s[:self.m, :]  # torch.Size([32, 16])
            s2 = ASD_s[self.m:, :]  # torch.Size([32, 16])

            q_bar1 = torch.cat([s1, z2], dim=1)
            q_bar2 = torch.cat([s2, z1], dim=1)
            q_bar = torch.cat([q_bar1, q_bar2], dim=0)
            q_1 = torch.cat([s1, z1], dim=1)
            q_2 = torch.cat([s2, z2], dim=1)
            q = torch.cat([q_1, q_2], dim=0)
            q_bar_score = (self.discriminator(q_bar) + .1) * .85  # +.1 * .85 so that it's 0<x<1
            q_score = (self.discriminator(q) + .1) * .85
            return q_score, q_bar_score

        def forward(self, ASD_input, NC_input):
            # print("ASD_input",ASD_input.shape)#torch.Size([64, 1, 64, 64, 64])
            # print("NC_input",NC_input.shape)#torch.Size([64, 1, 64, 64, 64])
            ASD_z_mean, ASD_z_log_var, ASD_z, ASD_out_shape = self.encoder_Z(ASD_input)
            ASD_s_mean, ASD_s_log_var, ASD_s, _ = self.encoder_S(ASD_input)
            NC_z_mean, NC_z_log_var, NC_z, _ = self.encoder_Z(NC_input)
            z = torch.cat([ASD_z, ASD_s], dim=1)
            ASD_decoder_out = self.decoder(z)
            zeros = torch.zeros_like(ASD_z)
            z_nc = torch.cat([NC_z, zeros], dim=1)
            NC_decoder_out = self.decoder(z_nc)
            return ASD_decoder_out, NC_decoder_out, ASD_z_mean, ASD_z_log_var, ASD_z, ASD_s_mean, ASD_s_log_var, ASD_s, NC_z_mean, NC_z_log_var, NC_z
class VAE(nn.Module):
    def __init__(self, input_shape=(64,64,64,1), latent_size=2, batch_size = 32, disentangle=False,
    gamma=1,
    kernel_size = 3,
    filters = 32,
    intermediate_dim = 128,
    nlayers = 2,
    bias=True):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.image_size, _, _, self.channels = input_shape
        self.kernel_size = kernel_size
        self.filters = filters
        self.intermediate_dim = intermediate_dim
        self.batch_size=batch_size
        self.latent_size = latent_size
        self.bias=bias
        self.nlayers=nlayers
        self.gamma=gamma
        self.disentangle=disentangle
        self.half_intermeidan=int(self.intermediate_dim/2)
        self.m = int(batch_size / 2)

        self.encoder_forward = nn.Sequential(
            nn.Conv3d(self.channels,96,kernel_size=self.kernel_size,stride=2,padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 192,kernel_size=self.kernel_size,stride=2,padding=(1,1,1),bias=self.bias),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Sequential(nn.Linear(786432, self.intermediate_dim,bias=self.bias),#786432
                                    nn.ReLU(inplace=True))
        self.mean_linear = nn.Linear(self.intermediate_dim, self.latent_size,bias=self.bias)
        self.log_var_linear = nn.Linear(self.intermediate_dim, self.latent_size,bias=self.bias)

        self.decoder_forward_linear = nn.Sequential(
            nn.Linear(self.latent_size , self.intermediate_dim,bias=self.bias),
            nn.ReLU(inplace=True),
            # nn.Linear(self.intermediate_dim,
            #           self.intermediate_dim * self.latent_size * self.latent_size * self.latent_size)
            nn.Linear(self.intermediate_dim,786432,bias=self.bias),
            nn.ReLU(inplace=True)
        )
        ##nn.Linear(self.intermediate_dim, self.ASD_out_shape[1]*self.ASD_out_shape[2]*self.ASD_out_shape[3]*self.ASD_out_shape[4])
        self.decoder_forward_up = nn.Sequential(
            nn.ConvTranspose3d(192, 192, kernel_size=self.kernel_size + 1, stride=2, padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(192, 96, kernel_size=self.kernel_size + 1, stride=2, padding=(1, 1, 1),bias=self.bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(96, 1, kernel_size=self.kernel_size, stride=1, padding=(1, 1, 1),bias=self.bias),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(nn.Linear(32, 1),
                                           nn.Sigmoid())
        self.loss_func = torch.nn.MSELoss()

    def encoder(self, X):
        s_out = self.encoder_forward(X)
        s_out_shape=s_out.shape
        s_out=s_out.view(s_out.size(0), -1)
        # print("s_out",s_out.shape)
        s_out=self.linear(s_out)
        s_mean =self.mean_linear(s_out)
        s_log_var = self.log_var_linear(s_out)
        s_s = self.reparameterization(s_mean, s_log_var)
        return s_mean, s_log_var,s_s,s_out_shape

    def decoder(self, z):
        z = self.decoder_forward_linear(z)
        # print(z.shape)
        b = z.shape[0]
        z = z.reshape(b, 192, 16, 16, 16)
        z = self.decoder_forward_up(z)
        return z

    def reparameterization(self, mean, log_var):
        # 从均值和方差分布中进行采样生成Z
        epsilon = torch.randn_like(log_var)
        z = mean + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, input,z_mean, z_log_var,z,decoder_out):
        # print("ASD_z_mean",ASD_z_mean.shape) # torch.Size([64, 16])
        # print("ASD_z_log_var",ASD_z_log_var.shape)#torch.Size([64, 16])
        reconstruction_loss = self.loss_func(torch.flatten(input, start_dim=1, end_dim=-1),
                                             torch.flatten(decoder_out, start_dim=1, end_dim=-1))
        # reconstruction_loss *= self.input_shape[0] * self.input_shape[1]  # * self.input_shape[2] * self.input_shape[3]
        kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)

        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5

        if self.disentangle:
            q_score, q_bar_score = self.forward_disentangle(z)
            tc_loss = torch.log(q_score / (1 - q_score))
            discriminator_loss = - torch.log(q_score) - torch.log(1 - q_bar_score)

            vae_loss = torch.mean(reconstruction_loss) + torch.mean(kl_loss) + self.gamma * torch.mean(tc_loss) + torch.mean(discriminator_loss)

            loss_metric = [reconstruction_loss, kl_loss, tc_loss, discriminator_loss]

            return vae_loss, loss_metric
        else:
            vae_loss = torch.mean(reconstruction_loss) + torch.mean(kl_loss)

            return vae_loss

    def forward_disentangle(self,z):

        z1 = z[:self.m, :self.half_intermeidan]#torch.Size([32, 16])
        z2 = z[self.m:, :self.half_intermeidan]#torch.Size([32, 16])
        s1 = z[:self.m, self.half_intermeidan:]#torch.Size([32, 16])
        s2 = z[self.m:, self.half_intermeidan:]#torch.Size([32, 16])

        q_bar1=torch.cat([s1, z2], dim=1)
        q_bar2=torch.cat([s2, z1], dim=1)
        q_bar=torch.cat([q_bar1,q_bar2],dim=0)
        q_1=torch.cat([s1, z1], dim=1)
        q_2=torch.cat([s2, z2], dim=1)
        q=torch.cat([q_1,q_2],dim=0)
        q_bar_score = (self.discriminator(q_bar) + .1) * .85  # +.1 * .85 so that it's 0<x<1
        q_score = (self.discriminator(q) + .1) * .85
        return q_score,q_bar_score

    def forward(self,input):
        z_mean, z_log_var,z,z_out_shape=self.encoder(input)
        decoder_out=self.decoder(z)
        return z_mean, z_log_var,z,z_out_shape,decoder_out

