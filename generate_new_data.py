import meshio
import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
from torchdiffeq import odeint

T=1
points=meshio.read("data/Stanford_Bunny.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)

reference_triangles=meshio.read("data/Stanford_Bunny.stl").cells_dict["triangle"]

x=torch.tensor(points,dtype=torch.float32)
y=x.clone()

BATCH_SIZE=1

hidden_dim=100
class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.nn1=nn.Sequential(nn.Linear(6,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))
        self.nn2=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,5))

    def forward(self,x,pos):
        x=torch.cat((x,pos),dim=2)
        x=x.reshape(-1,6)
        x=self.nn1(x)
        x=x.reshape(BATCH_SIZE,-1,hidden_dim)
        x=torch.mean(x,dim=1)
        x=self.nn2(x)
        return x

class ScoreFunction(nn.Module):
    def __init__(self, latent_size):
        super(ScoreFunction, self).__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(nn.Linear(latent_size+1, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, self.latent_size))
        self.T=T
    
    def beta(self,t):
        return torch.tensor(1.0)
    
    def int_beta(self,t):
        return torch.tensor(t)
    
    def weight(self,t):
        return torch.exp(-t)

    def drift_coef(self,t):
        return -1/2*self.beta(t)

    def diffusion(self,t):
        return torch.sqrt(self.beta(t))

    def mean_p_0t(self,x0,t):
        return x0*torch.exp(-1/2*self.int_beta(t))
    
    def std_p_0t(self,x0,t):
        return torch.max(1-torch.exp(-self.int_beta(t)),torch.tensor(0.0001))

    def der_likelihood(self,x,x0,t):
        return -1/self.std_p_0t(x0,t)*(x-self.mean_p_0t(x0,t))

    def diff_eq(self,t,x):
        return -1/2*self.beta(t)*(x+self.forward(x,t))

    def sample(self,num_samples):
        t=torch.linspace(0,1,100).flip(0)
        x=torch.randn(num_samples,self.latent_size)
        y=odeint(self.diff_eq,x,t)
        return y[-1]

    def forward(self, x, t):
        if t.ndim<2:
            t=torch.tensor(t)
            t=t.unsqueeze(0)
            t=t.unsqueeze(0)
            t=t.repeat(x.shape[0],1)
        tmp=torch.cat((x,t),1)
        return self.model(tmp)


class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim=latent_dim
        self.model=nn.Sequential(nn.Linear(3+latent_dim,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Linear(100,100),nn.BatchNorm1d(100),nn.Linear(100,3))


    def forward(self,latent,pos):
        pos=pos.reshape(BATCH_SIZE,-1,3)
        latent=latent.reshape(-1,1,self.latent_dim).repeat(1,pos.shape[1],1)
        x=torch.cat((latent,pos),dim=2)
        x=x.reshape(-1,3+self.latent_dim)
        x=self.model(x)
        x=x.reshape(BATCH_SIZE,-1,3)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder=Encoder(latent_dim)
        self.decoder=Decoder(latent_dim)

    def forward(self,batch):
        x,pos=batch
        latent=self.encoder(x,pos)
        x=self.decoder(latent,pos)
        return x,latent



points_mesh=torch.tensor(points,dtype=torch.float32)
model=AutoEncoder(latent_dim=5)


model=torch.load("model_pyg.pt")



class Generator(nn.Module):
    def __init__(self,latent_dim):
        super(Generator, self).__init__()
        self.model=torch.load("model_pyg.pt")
        self.model.eval()
        self.latent_dim=latent_dim
        self.sde=torch.load("score.pt")
        self.sde.eval()

    def forward(self, pos):
        x=self.sde.sample(1)
        return self.model.decoder(x,pos)

model=Generator(5)
model.eval()

all_gen=np.zeros((600,len(points),3))
with torch.no_grad():
    for i in range(600):
        all_gen[i]=model(points_mesh).detach().numpy()



np.save("all_points_gen.npy",all_gen)
