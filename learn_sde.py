import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
T=1
from torchdiffeq import odeint
# Define the SDE





# Load data
import numpy as np
latent=np.load("latent.npy")
latent=latent.reshape(latent.shape[0],-1)
latent_size=latent.shape[1]
latent=torch.tensor(latent).float()


    

class ScoreFunction(nn.Module):
    def __init__(self, latent_size):
        super(ScoreFunction, self).__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(nn.Linear(latent_size+1, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, self.latent_size))
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
            print(x.shape)
            t=torch.tensor(t)
            t=t.unsqueeze(0)
            t=t.unsqueeze(0)
            t=t.repeat(x.shape[0],1)
        tmp=torch.cat((x,t),1)
        return self.model(tmp)
    



dataset=torch.utils.data.TensorDataset(latent)
loader=torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)



model=ScoreFunction(latent_size)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
loss_list = []
for epoch in range(num_epochs):
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        x0=x[0]
        t=torch.rand(x0.shape[0],1)*T
        x_t=model.mean_p_0t(x0,t)+torch.sqrt(model.diffusion(t))*torch.randn(x0.shape[0],latent_size)
        loss = torch.mean(model.weight(t)*torch.linalg.norm(model(x_t,t)-model.der_likelihood(x_t,x0,t),dim=1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.3f}'.format(
            epoch+1, num_epochs, i+1, len(loader), loss.item()))
    if loss.item()<100:
        loss_list.append(loss.item())
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()
torch.save(model, "score.pt")
