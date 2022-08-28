from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from model import encoder
from model import decoder
import numpy as np

def train_function(model1,model2,loader,opt,sr,criterion,device,c):
  model1.train()
  model2.train()
  loop = tqdm(loader,leave=True)
  loss_list = []
  for idx,(data) in enumerate(loop):
    data = data.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    with torch.cuda.amp.autocast():
      encode = model1(data)
      decode = model2(encode)
      loss = criterion(decode,data)
      opt.zero_grad()
      loss.backward()
      opt.step()
      loss_list.append(loss.cpu().detach().clone().numpy())
    total_diff =sum(loss_list)/len(loss_list)
  print(f'epoch : {c}  average loss : {total_diff}')
  return total_diff


def train(feature_dim,encoder,decoder,dataset,lr,device,epoch,load_weight = False):
  if load_weight == True:
    #if you use cpu for inference, pre trained weight is trained on cuda, if use gpu inference
    #change 'cpu' to 'cuda'
    model1 = encoder(feature_dim)
    model2 = decoder(feature_dim)
    model1.load_state_dict(torch.load('model1_encoder.pt',map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load('model2_decoder.pt',map_location=torch.device('cpu')))

  else:
    model1 = encoder(feature_dim)
    model2 = decoder(feature_dim)
  dataset = dataset.astype('float32')

  opt = optim.Adam(list(model1.parameters())+list(model2.parameters()), lr=lr,betas=(0.5,0.999),)
  scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
  criterion = nn.MSELoss()
  train_loader = DataLoader(dataset,batch_size=64,shuffle=True,num_workers=2,pin_memory=True)
  device= device
  counter = 0 

  for epoch in range(epoch):
    train_function(model1,model2,train_loader,opt,scheduler,criterion,device,c=counter)
    counter +=1

if __name__ == "__main__":
  concat = np.load('concat.npy')
  train(feature_dim=219,encoder=encoder,decoder=decoder,dataset=concat,lr=0.0001,device='cpu',epoch=1,load_weight=True)

    

