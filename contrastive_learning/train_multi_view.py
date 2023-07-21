import os
import math
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import simsiam.loader
import simsiam.builder

from simsiam_dataset import multi_view_dataset

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def calculate_loss(model, loss_function, img_1, img_2):
    img_1 = img_1.cuda()
    img_2 = img_2.cuda()
    p1, p2, z1, z2 = model(x1=img_1, x2=img_2)
    if loss_function == "MSE":
        loss = criterion(p1, z2).mean() * 0.5 + criterion(p2, z1).mean() * 0.5 # 
    else:
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5 # 
    return loss


parser = argparse.ArgumentParser(description='Contrastive Learning of DRR')
parser.add_argument('-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('-batch_size', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=300, type=int, help='training epochs')
parser.add_argument('-model', default="resnet50", type=str, help='backbone model')
parser.add_argument('-loss', default="MSE", type=str, help='loss function')
parser.add_argument('-img_size', default=256, type=int, help='img size')
parser.add_argument('-n_views', default=10, type=int, help='number of views')
parser.add_argument('-weights', default="None", type=str, help='weights: NONE, IMAGENET1K_V1, IMAGENET1K_V2')
parser.add_argument('-log_path', default="simsiam", type=str, help='the path of the log') # simsiam_MSE_256_IMAGENET1K_V2

########
# python contrastive_learning/simsiam-main/train_multi_view.py -lr 0.05 -batch_size 64 -epochs 300 -n_views 10 -model "resnet50" -loss "cos" -weights "IMAGENET1K_V1"
########

if __name__=="__main__":
    args = parser.parse_args()

    base_path = "/public_bme/data/wuhan/spine/dataset/drr/"
    
    lr = args.lr
    batch_size = args.batch_size
    init_lr = lr * batch_size / 256
    epochs = args.epochs
    
    model = simsiam.builder.SimSiam(models.__dict__[args.model], 2048, 512, args.weights)
    model.cuda()

    log_path = os.path.join("logs/simsiam/" + str(args.n_views) + "views/", str(args.batch_size), args.log_path+"_"+args.loss+"_"+str(args.img_size)+"_"+args.weights)
    checkpoint_path = os.path.join("checkpoints/simsiam/" + str(args.n_views) + "views/", str(args.batch_size), args.log_path+"_"+args.loss+"_"+str(args.img_size)+"_"+args.weights)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_path, str(args.epochs)))
    if args.loss == "MSE":
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.CosineSimilarity(dim=1).cuda()

    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    
    
    mean_std = [[0.456,0.456,0.456], [0.224,0.224,0.224]]
    train_data = multi_view_dataset(mode="train", mean_std=mean_std, n_views=args.n_views)
    test_data = multi_view_dataset(mode="eval", mean_std=mean_std, n_views=args.n_views)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    print(f'[Summary] lr is {lr}, training epoch is {epochs}, batch size is {batch_size}, model is {args.model}, loss is {args.loss}, log path is {log_path}')
    print(f'-------- start training --------')
    
    for epoch in tqdm(range(epochs)):
        adjust_learning_rate(optimizer, init_lr, epoch, epochs)
        
        tqdm.write(f'--------[{epoch+1}/{epochs}]--------')
        model.train()
        running_loss = 0.0
        for i,images in enumerate(train_loader):
            img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2 = images
            # img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2 = img_1.cuda(), img_2.cuda(), aug_1_1.cuda(), aug_1_2.cuda(), aug_2_1.cuda(), aug_2_2.cuda()
  
            loss_img = calculate_loss(model, args.loss, img_1, img_2)
            # loss_aug = calculate_loss(model, args.loss, aug_1_1, aug_2_1)
            # loss_aug_1 = calculate_loss(model, args.loss, aug_1_1, aug_1_2)
            # loss_aug_2 = calculate_loss(model, args.loss, aug_2_1, aug_2_2)
            loss = loss_img #(loss_img + loss_aug_1 + loss_aug_2)/3
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('Train Loss', running_loss / (i+1), epoch+1)
        tqdm.write(f'Train loss is {running_loss/(i+1):.5f} lr is {optimizer.param_groups[0]["lr"]}')


        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i,images in enumerate(test_loader):
                img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2 = images
                # img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2 = img_1.cuda(), img_2.cuda(), aug_1_1.cuda(), aug_1_2.cuda(), aug_2_1.cuda(), aug_2_2.cuda()
    
                loss_img = calculate_loss(model, args.loss, img_1, img_2)
                # loss_aug = calculate_loss(model, args.loss, aug_1_1, aug_2_1)
                # loss_aug_1 = calculate_loss(model, args.loss, aug_1_1, aug_1_2)
                # loss_aug_2 = calculate_loss(model, args.loss, aug_2_1, aug_2_2)
                loss = loss_img #(loss_img + loss_aug_1 + loss_aug_2)/3
                running_loss += loss.item()
        

        writer.add_scalar('Test Loss', running_loss / (i+1), epoch+1)
        tqdm.write(f'Test loss is {running_loss/(i+1):.5f} lr is {optimizer.param_groups[0]["lr"]}')
        if (epoch+1)%10 == 0:
            checkpoint = {
                        "net": model.state_dict(),
                        "epoch": epoch,
                        }
            torch.save(checkpoint,checkpoint_path + str(epoch+1)+".pth")
