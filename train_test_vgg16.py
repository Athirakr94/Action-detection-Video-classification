import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import get_video_backend
from torchvision.models import vgg16 
from torchvision import transforms
import os
from tqdm.auto import tqdm
import numpy as np
import time
import av
import random
import transforms as T

print(f"PyAV version -- {av.__version__}")

SEED = 491
torch.manual_seed(SEED)

from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# set up a sample pytorch class to check if it works
class Model(nn.Module):
  def __init__(self):
      super(Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=False).children())[:-1])
      self.fc1 = nn.Linear(512, 51)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2)
      print("size after pretrained model ", out.size())
      out = torch.log_softmax(self.fc1(out), dim=1)
      return out


#check = MultiLayerBiLSTMClassifier().cuda()
#out = check(torch.randn(16, 3 , 8, 112,112).cuda())
#print(out.size())

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_acc(self, val, n=1):
        self.val = val/n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# define model using PyTorch
class VideoRecog_Model(nn.Module):
  def __init__(self):
      super(VideoRecog_Model, self).__init__()
      self.base_model = nn.Sequential(*list(vgg16(pretrained=True).children())[:-1])
      self.fc1 = nn.Linear(512, 51)
      self.fc2 = nn.Linear(51, 51)
      self.dropout = nn.Dropout2d(0.3)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2) # output of base model is bs x 512 x 1 x 1 x 1
      out = F.relu(self.fc1(out))
      out = self.dropout(out)
      out = torch.log_softmax(self.fc2(out), dim=1)
      return out

def train(config, model, loader, optimizer, epoch):
    model.train()
    config = {}
    config['log_interval'] = 100
    correct = 0
    total_loss = 0.0
    flag = 0
    Loss, Acc = AverageMeter(), AverageMeter()
    start = time.time()
    for batch_id, data in enumerate(loader):
        data, target = data[0], data[-1]
        # print("here")

        if torch.cuda.is_available():
           data = data.cuda()
           target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        Loss.update(loss.item(), data.size(0))

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_corrects = pred.eq(target.view_as(pred)).sum().item()
        correct += num_corrects

        Acc.update_acc(num_corrects, data.size(0))

        # if flag!= 0 and batch_id%config['log_interval'] == 0:
        #    print('Train Epoch: {} Batch [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.0f})%'.format(
        #         epoch, batch_id * len(data), len(loader.dataset),
        #         100. * batch_id / len(loader), Loss.avg, correct, Acc.count, 100. * Acc.avg))
        flag = 1

    #total_loss /= len(loader.dataset)
    print('Train Epoch: {} Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         epoch, Loss.avg, correct, Acc.count, 100. * Acc.avg ))
    print(f"Takes {time.time() - start}")
    return Loss.avg,(correct/Acc.count )* 100

def test(config, model, loader, text='Validation'):
    model.eval()
    correct = 0
    total_loss = 0.0
    Loss, Acc = AverageMeter(), AverageMeter()
    with torch.no_grad():
         for batch_id, data in enumerate(loader):
             data, target = data[0], data[-1]

             if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

             output = model(data)
             loss = F.nll_loss(output, target)
             total_loss += loss.item()

             Loss.update(loss.item(), data.size(0))

             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
             num_corrects = pred.eq(target.view_as(pred)).sum().item()
             correct += num_corrects

             Acc.update_acc(num_corrects, data.size(0))

    total_loss /= len(loader.dataset)
    print(text + ' Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         Loss.avg, correct, Acc.count , 100. * Acc.avg ))
    return Loss.avg,(correct/Acc.count )* 100

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
total_val_samples = round(val_split * total_train_samples)

print(f"number of train samples {total_train_samples}")
print(f"number of validation samples {total_val_samples}")
print(f"number of test samples {len(hmdb51_test)}")

bs = 8
lr = 1e-4
gamma = 0.7
total_epochs = 30
config = {}
num_workers = 0

kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
#kwargs = {'num_workers':num_workers}
#kwargs = {}

hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                       total_val_samples])

#hmdb51_train_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_val_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_test.video_clips.compute_clips(16, 1, frame_rate=30)

#train_sampler = RandomClipSampler(hmdb51_train_v1.video_clips, 5)
#test_sampler = UniformClipSampler(hmdb51_test.video_clips, 5)

train_loader = DataLoader(hmdb51_train_v1, batch_size=bs, shuffle=True, **kwargs)
val_loader   = DataLoader(hmdb51_val_v1, batch_size=bs, shuffle=True, **kwargs)
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

model = VideoRecog_Model()
# model = MultiLayerBiLSTMClassifier(input_size, hidden_size, 2, num_classes).cuda()
if torch.cuda.is_available():
   model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
train_losses=[]
test_losses=[]
train_accuracies=[]
test_accuracies=[]
print("Launching Action Recognition Model training")
for epoch in range(1, total_epochs + 1):
    train_loss, train_accuracy=train(config, model, train_loader, optimizer, epoch)
    val_loss,val_accuracy=test(config, model, val_loader, text="Validation")
    # scheduler.step()
    with open("vgg_log1.txt", "a") as myfile:
        text="Epoch "+str(epoch)+","+str(train_loss)+","+str(train_accuracy)+","+str(val_loss)+","+str(val_accuracy)+"\n"
        myfile.write(text)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(val_accuracy)
    train_losses.append(train_loss)
    test_losses.append(val_loss)
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
plt.savefig("Plotvgg.jpg")
test_loss,test_accuracy=test(config, model, test_loader, text="Test")
print(test_loss,test_accuracy)

