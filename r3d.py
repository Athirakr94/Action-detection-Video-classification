import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import get_video_backend
from torchvision.models.video import r3d_18 
from torchvision import transforms
from matplotlib import pyplot as plt
import os
from tqdm.auto import tqdm
import numpy as np
import time
import av
import random
import transforms as T
# Datasets and Dataloaders for model training ..

val_split = 0.05
num_frames = 16 # 16
clip_steps = 50
num_workers = 8
pin_memory = True
train_tfms = torchvision.transforms.Compose([
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((128, 171)),
                                 T.RandomHorizontalFlip(),
                                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((112, 112))
                               ])
test_tfms =  torchvision.transforms.Compose([
                                             T.ToFloatTensorInZeroOne(),
                                             T.Resize((128, 171)),
                                             T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                             T.CenterCrop((112, 112))
                                             ])
hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=True,
                                                transform=train_tfms, num_workers=num_workers)


hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=test_tfms, num_workers=num_workers)

total_train_samples = len(hmdb51_train)
total_test_samples = len(hmdb51_test)
bs=4
print(f"number of train samples {total_train_samples}")
print(f"number of test samples {len(hmdb51_test)}")
train_loader = DataLoader(hmdb51_train, batch_size=bs, shuffle=True)
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False)
print("data loaded")
# define model using PyTorch
class Model(nn.Module):
  def __init__(self):
      super(Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=True).children())[:-1])
      self.fc1 = nn.Linear(512, 51)
      self.fc2 = nn.Linear(51, 51)
      self.dropout = nn.Dropout2d(0.3)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2) # output of base model is bs x 512 x 1 x 1 x 1
      out = F.relu(self.fc1(out))
      out = self.dropout(out)
      out = torch.log_softmax(self.fc2(out), dim=1)
      return out
class Model_Old(nn.Module):
  def __init__(self):
      super(Model_Old, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=True).children())[:-1])
      self.fc1 = nn.Linear(512, 51)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2)
      # print("size after pretrained model ", out.size())
      out = torch.log_softmax(self.fc1(out), dim=1)
      return out
model=Model().cuda()
lr=1e-3
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train( model, loader, optimizer, epoch,samples):
    model.train()
    config = {}
    config['log_interval'] = 100
    correct = 0
    total_loss = 0.0
    flag = 0
   
    start = time.time()
    for batch_id, data in enumerate(loader):
        data, target = data[0], data[-1]
        # print("here")

        if torch.cuda.is_available():
           data = data.cuda()
           target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_corrects = pred.eq(target.view_as(pred)).sum().item()
        correct += num_corrects
    total_loss=total_loss/samples
    print('Train Epoch: {} Loss: {:.6f}  Accuracy: {:.6f} '.format(epoch, total_loss,  (correct/samples)* 100))
    print(f"Takes {time.time() - start}")
    return total_loss,  (correct/samples)* 100
def test( model, loader,samples):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
         for batch_id, data in enumerate(loader):
             data, target = data[0], data[-1]

             if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

             output = model(data)
             loss = criterion(output, target)
             total_loss += loss.item()

             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
             num_corrects = pred.eq(target.view_as(pred)).sum().item()
             correct += num_corrects

    total_loss=total_loss/(samples)
    print('Test Loss: {:.6f}  Accuracy: {:.6f} '.format(total_loss,  (correct/samples)* 100))
    return total_loss,   (correct/samples)* 100
print("Launching Action Recognition Model training")
train_losses=[]
test_losses=[]
train_accuracies=[]
test_accuracies=[]
total_epochs=50
for epoch in range(1, total_epochs + 1):
    train_loss, train_accuracy=train( model, train_loader, optimizer, epoch,total_train_samples)
    val_loss,val_accuracy=test( model, test_loader,total_test_samples)
    # scheduler.step()
    with open("r3d_res_dropout.txt", "a") as myfile:
        text="Epoch "+str(epoch)+","+str(train_loss)+","+str(train_accuracy)+","+str(val_loss)+","+str(val_accuracy)+"\n"
        myfile.write(text)


    train_accuracies.append(train_accuracy)
    test_accuracies.append(val_accuracy)
    train_losses.append(train_loss)
    test_losses.append(val_loss)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join("r3d_res", str(epoch)+'.pth'))
    if epoch%5==0:
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.plot(train_losses, label='Training Loss')
      plt.plot(test_losses, label='Testing Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.subplot(1, 2, 2)
      plt.plot(train_accuracies, label='Training Accuracy')
      plt.plot(test_accuracies, label='Testing Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.savefig("Plotr3d_drop.jpg")
      plt.close()
