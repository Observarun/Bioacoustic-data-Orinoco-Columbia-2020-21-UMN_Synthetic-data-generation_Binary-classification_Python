# Variational autoencoder model for generating synthetic spectrogram images 



import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import LeakyReLU
from torch.nn import MaxPool2d
import torch.nn.functional as F



class VariAutoEnco(nn.Module):
  def __init__(self, latent_dim):
    super(VariAutoEnco,self).__init__() #constructor of the super class called
    
    self.enco_conv = nn.Sequential(
        # Is understood that filter has dimensions 5*5*in_channels. out_channels = 16 = # filters.
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same', stride=1, bias=False),
        # 'same' padding => spatial resolution after Conv2d layer is the same as i/p.
        # O/p from this layer is 288*432*16.
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # O/p from pooling layer is 144*216*16

        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 144*216*64
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # O/p is 72*108*64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 72*108*128
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # O/p is 36*54*128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 36*54*256
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # O/p is 18*27*256

        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 18*27*256
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        # O/p is 9*13*256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 9*13*512
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        # O/p is 5*7*1024

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding='same', stride=1, bias=False),
        # O/p is 5*7*1024
        nn.ReLU(),
        nn.BatchNorm2d(1024),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # O/p is 3*4*1024
    )

    self.mu_fc = nn.Sequential(
        nn.Linear(1024*3*4,latent_dim), #fully connected layer
        nn.ReLU(),
        nn.Dropout(.2)
    )

    self.log_var_fc = nn.Sequential(
        nn.Linear(1024*3*4,latent_dim), #fully connected layer
        nn.ReLU(),
        nn.Dropout(.2)
    )

    self.deco_fc = nn.Sequential(
        nn.Linear(latent_dim,1024*3*4), #fully connected layer
        nn.ReLU(),
        nn.Dropout(.2)
    )

    self.latent_dim = latent_dim

    self.deco_conv = nn.Sequential(
        #1024*3*4
        nn.ConvTranspose2d(1024, 512, kernel_size=(3,3), padding=1, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        # 512*5*7
        
        nn.ConvTranspose2d(512, 256, kernel_size=(3,3), padding=1, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        # 256*9*13
        
        nn.ConvTranspose2d(256, 256, kernel_size=(4,5), padding=1, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        # 256*18*27

        nn.ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        # 128*36*54

        nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        #128*72*108

        nn.ConvTranspose2d(64, 16, kernel_size=2, padding=0, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        #16*144*216

        nn.ConvTranspose2d(16, 3, kernel_size=2, padding=0, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(3),
        # 3*288*432

        nn.Sigmoid()
    )
  
  def encode(self,x):
    x = self.enco_conv(x)
    x = x.view(x.shape[0], -1) #flattening
    mu = self.mu_fc(x)
    log_var = self.log_var_fc(x)
    return mu, log_var
        
  def reparameterise(self, mu, log_var):
    std = torch.exp(.5*log_var)
    eps = torch.randn_like(std)
    z = mu + eps*std
    return z

  def decode(self, z):
    x = self.deco_fc(z)
    x = x.view(x.shape[0], 1024, 3, 4) #unflattening
    recon = self.deco_conv(x)
    return recon

  def forward(self, x):
    # connection between different layers
    mu, log_var = self.encode(x)
    z = self.reparameterise(mu, log_var)
    recon = self.decode(z)
    return recon, mu, log_var

