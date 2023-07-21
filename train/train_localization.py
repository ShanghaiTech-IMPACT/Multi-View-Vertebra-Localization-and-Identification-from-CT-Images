import argparse 
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn.functional as F
import sys
sys.path.append(".")
from models.ResUnet import UNetWithResnet50Encoder
from models.drr_net import drr_net
from models.deeplabv3 import *
from data.drr_dataset import drr_dataset
# from data.multichannel_heatmap import drr_dataset


class log_writer():
    def __init__(self, txt_name = ""):
        super().__init__()
        self.name = txt_name
    def write(self,info):
        writer = open(self.name,"a+")
        data = info + "\n"
        writer.writelines(data)
        print(info)
        writer.close()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Localization training')
parser.add_argument('-lr',                          default=0.0001,     type=float, help='learning rate')
parser.add_argument('-batch_size',                  default=1,          type=int,   help='batch size')
parser.add_argument('-epochs',                      default=120,        type=int,   help='training epochs')
parser.add_argument('-eval_epoch',                  default=4,          type=int,   help='evaluation epoch')
parser.add_argument('-log_path',                    default="logs",     type=str,   help='the path of the log') 
parser.add_argument('-log_inter',                   default=50,         type=int,   help='log interval')
parser.add_argument('-read_params',                 default=False,      type=bool,  help='if read pretrained params')
parser.add_argument('-params_path',                 default="",         type=str,   help='the path of the pretrained model')
parser.add_argument('-basepath',                    default="",         type=str,   help='base dataset path')
parser.add_argument('-augmentation',                default=False,      type=bool,  help='if augmentation')
parser.add_argument('-num_view',                    default=10,         type=int,   help='the number of views')

if __name__=="__main__":
    setup_seed(33)
    args = parser.parse_args()
    n_views = args.num_view
    log_path = args.log_path
    log_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer = SummaryWriter(log_path+log_name)
    txt_name = log_path + log_name + ".txt"
    txt_writer = log_writer(txt_name)
    base_train_path = f"{args.basepath}/train/{n_views}_views" 
    base_test_path = f"{args.basepath}/test/{n_views}" 
    save_path = log_path 
    epochs = args.epochs
    base_lr = args.lr
    eval_epoch = args.eval_epoch
    log_inter = args.epochs
    compare = []
    model = UNetWithResnet50Encoder(1, pretrained=args.read_params)
    if args.read_params:
        state_dict = torch.load(args.params_path)
        model.load_state_dict(state_dict['net'])
    mdeol = model.cuda()
    model.train()
    loss_func = nn.MSELoss()
    loss_func = loss_func.cuda()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    schedule = MultiStepLR(optimizer, milestones=[epochs//4, epochs//4*2, epochs//4*3], gamma=0.1)
    txt_writer.write("-"*8 + "reading data" + "-"*8)
    train = drr_dataset(drr_path=base_train_path, mode="train", if_identification=False, n_views=n_views)
    test = drr_dataset(drr_path=base_test_path, mode="test", if_identification=False, n_views=n_views)
    
    trainset = DataLoader(dataset=train, batch_size=1, shuffle = False)
    testset = DataLoader(dataset=test, batch_size=1, shuffle = False)
    txt_writer.write("-"*8 + "start localization training" + "-"*8)
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(trainset):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
  
            if i % log_inter == (log_inter-1): 
                txt_writer.write(f'[*]epoch {epoch+1} [{i}/{len(trainset)}] loss is {running_loss/(i+1):.8f}')
        writer.add_scalar('Train Loss', running_loss / (i+1), epoch)
        schedule.step()
        epoch_lr = optimizer.param_groups[0]['lr']
        txt_writer.write(f'[*]Training epoch {epoch+1} loss : {running_loss/(i+1):.8f}, lr is {epoch_lr}')
        running_loss = 0.0

        if epoch%eval_epoch == eval_epoch-1:
            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for (data, target) in testset:
                    data, target = data.float(), target.float()
                    data = data.cuda()
                    target = target.cuda()
                    output = model(data)
  
                    loss = loss_func(output, target)
                    test_loss += loss.item()

            writer.add_scalar('Test Loss', test_loss / len(testset), epoch)

            now = time.time()
            period = str(datetime.timedelta(seconds=int(now-start_time)))
            txt_writer.write(f'[*]Test finish, test epoch {epoch+1} loss : {test_loss / len(testset):.8f} , training time is {period}')
            model.train()
            txt_writer.write("Save params to " + save_path+str(epoch+1)+".pth")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': schedule.state_dict()
                }
            torch.save(checkpoint,save_path+str(epoch)+".pth")