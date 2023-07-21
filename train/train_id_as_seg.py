import argparse
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR,ReduceLROnPlateau

import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append(".")
from dataset.id_as_seg_dataset import id_as_seg_dataset
from models.ResUnet import UNetWithResnet50Encoder

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def compute_sequence_loss(output, target, class_num = 24, inference = False):
    output = output.squeeze()
    output = torch.softmax(output, dim=0)
    id_list = torch.unique(target)[1:]
    for i,id in enumerate(id_list):
        index = (target==id)
        id_output = output * index
        
        for j in range(id_output.shape[0]):
            if j == 0:
                tmp = id_output[j][index[0]]
                tmp = tmp.unsqueeze(0)
            else:
                tmp = torch.cat((tmp, id_output[j][index[0]].unsqueeze(0)), dim=0)
        if i == 0:
            probability = torch.mean(tmp, dim=1)
            probability = probability.unsqueeze(0)
        else:
            probability = torch.cat((probability, torch.mean(tmp, dim=1).unsqueeze(0)), dim = 0)

    for i in range(len(id_list)):
        for j in range(class_num):
            if j > 1 and i > 0:
                probability[i][j] = max(probability[i-1][j-2]*0.1 + probability[i][j], probability[i-1][j-1]*0.8 + probability[i][j], probability[i-1][j]*0.1 + probability[i][j])
    if inference:
        return probability
    else:
        return 1 - torch.max(probability)/(len(id_list)*0.8)


def get_acc(out,gt_label, index):
    out = torch.argmax(torch.softmax(out, dim = 1), dim = 1)
    acc = (out[index]==gt_label[index]).sum()/(gt_label > 0).sum()
    return acc

def augmentation(img, target, transform):
    img = img.squeeze().numpy()
    target = target.squeeze().numpy()
    transformed  = transform(image = img[0], mask = target)
    img = transformed["image"]
    target = transformed["mask"]

    img = img[np.newaxis,:,:]
    img = np.concatenate((img,img,img),axis=0)
    img = torch.FloatTensor(img).unsqueeze(0)
    target = torch.LongTensor(target).unsqueeze(0)
    return img.cuda(), target.cuda()

def get_model(model_name, number_class, use_contrastive_learning, contrastive_learning_path, read_params, params_path):
    if model_name == "fcn":
        if use_contrastive_learning:
            print(f'-----load contrastive learning params from {contrastive_learning_path}-----')
            model = models.segmentation.fcn_resnet50(weights=None)
            model.classifier[-1] = nn.Conv2d(512,number_class,1,1)
            pretrained_dict = torch.load(contrastive_learning_path)['net']
            model_dict = model.state_dict()
            common_dict = {}
            for k in pretrained_dict.keys():
                name = "backbone" + k[7:]
                if name in model_dict.keys():
                    common_dict[name] = pretrained_dict[k]
            model_dict.update(common_dict)
            model.load_state_dict(model_dict)
        elif read_params:
            print(f'-----load trained params from {params_path}-----')
            model = models.segmentation.fcn_resnet50(num_classes=number_class, weights=None)
            pretrained_dict = torch.load(params_path)['net']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            model.classifier[-1] = nn.Conv2d(512,number_class,1,1)
    else:
        if use_contrastive_learning:
            print(f'-----load contrastive learning params from {contrastive_learning_path}-----')
            model = UNetWithResnet50Encoder(1, number_class)
            pretrained_dict = torch.load(contrastive_learning_path)['net']
            model_dict = model.state_dict()
            common_dict = {}
            for k in pretrained_dict.keys():
                name = "backbone" + k[7:]
                if name in model_dict.keys():
                    common_dict[name] = pretrained_dict[k]
            model_dict.update(common_dict)
            model.load_state_dict(model_dict)
        elif read_params:
            print(f'-----load trained params from {params_path}-----')
            model = UNetWithResnet50Encoder(1, number_class)
            pretrained_dict = torch.load(params_path)['net']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
             model = UNetWithResnet50Encoder(1, number_class)
    return model

parser = argparse.ArgumentParser(description='Identification training')
parser.add_argument('-lr',                          default=0.0001,     type=float, help='learning rate')
parser.add_argument('-batch_size',                  default=1,          type=int,   help='batch size')
parser.add_argument('-epochs',                      default=100,        type=int,   help='training epochs')
parser.add_argument('-eval_epoch',                  default=1,          type=int,   help='evaluation epoch')
parser.add_argument('-log_path',                    default="logs",     type=str,   help='the path of the log') 
parser.add_argument('-log_inter',                   default=50,         type=int,   help='log interval')
parser.add_argument('-use_contrastive_learning',    default=True,       type=bool,  help='if use contrastive_learning as pretrained')
parser.add_argument('-contrastive_learning_path',   default="",         type=str,   help='the path of the pretrained model')
parser.add_argument('-read_params',                 default=False,      type=bool,  help='if read pretrained params')
parser.add_argument('-params_path',                 default="",         type=str,   help='the path of the pretrained model')
parser.add_argument('-basepath',                    default="",         type=str,   help='base dataset path')
parser.add_argument('-augmentation',                default=False,      type=bool,  help='if augmentation')
parser.add_argument('-num_class',                   default=25,         type=int,   help='the number of class')
parser.add_argument('-model_name',                  default="",         type=str,   help='ResUnet or FCN')



if __name__=="__main__":
    setup_seed(33)
    args = parser.parse_args()
    contrastive_learning = args.use_contrastive_learning
    contrastive_learning_path = args.contrastive_learning_path
    read_params = args.read_params
    params_path = args.params_path

    if_deformation = args.augmentation
    transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.3),
                         ])

    epochs = args.epochs
    batch_size = args.batch_size
    number_class = args.num_class
    eval_epoch = args.eval_epoch
    log_inter = args.log_inter
    base_lr = args.lr
    save_name = str(batch_size)
    save_path = args.log_path
    writer = SummaryWriter(save_path+save_name)

    base_train_path = args.basepath + "train/"
    base_test_path = args.basepath + "test/"
    train = id_as_seg_dataset(drr_path=base_train_path, mode="train", if_deformation=if_deformation, transform=None)
    test = id_as_seg_dataset(drr_path=base_test_path, mode="test",if_deformation=False)
    trainset = DataLoader(dataset=train, batch_size=batch_size, shuffle = False)
    testset = DataLoader(dataset=test, batch_size=1, shuffle = False)

    start_epoch = 0
    model = get_model(args.model_name, number_class, contrastive_learning, contrastive_learning_path, read_params, params_path)
    model = model.cuda()

    loss_func_1 = nn.CrossEntropyLoss(reduction = 'none')
    loss_func_1 = loss_func_1.cuda()
 
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    schedule = MultiStepLR(optimizer, milestones=[epochs//3, epochs//3*2], gamma=0.1)

    print(f'save to {save_path}')
    for epoch in tqdm(range(start_epoch, epochs)):
        tqdm.write(f"----------{epoch}----------")
        model.train()
        training_CE_loss = 0.0
        training_sequence_loss = 0.0
        training_all_loss = 0.0
        acc = 0
        for i, (data, target) in enumerate(trainset):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.long().cuda()
            output = model(data)['out']
            
            dg_index = (target > 0)
            ce_loss = loss_func_1(output, target)[dg_index].mean()
            sequence_loss = compute_sequence_loss(output, target)
            loss = ce_loss + sequence_loss

            loss.backward()
            optimizer.step()

            training_sequence_loss += sequence_loss.item()
            training_CE_loss += ce_loss.item()
            training_all_loss += loss.item()
            acc += get_acc(output,target, dg_index)
            
            if(i%log_inter == log_inter - 1):
                tqdm.write(f"[{i+1}/{len(trainset)}] train loss: {training_all_loss/(i+1):.8f}, CE loss: {training_CE_loss/(i+1):.8f}, seq loss: {training_sequence_loss/(i+1):.8f}" )
        tqdm.write(f"[*]train finish, all loss is: {training_all_loss/(i+1):8f}, CE loss: {training_CE_loss/(i+1):.8f}, seq loss: {training_sequence_loss/(i+1):.8f}, acc is {100*acc/(i+1):.3f}%, lr is {optimizer.param_groups[0]['lr']}")
        writer.add_scalar('Training All Loss', training_all_loss / (i+1), epoch)
        writer.add_scalar('Training CE Loss', training_CE_loss / (i+1), epoch)
        writer.add_scalar('Training Sequence Loss', training_sequence_loss / (i+1), epoch)
        writer.add_scalar('Training Acc', acc/(i+1), epoch)
        schedule.step()

        if epoch%eval_epoch == eval_epoch-1:
            model.eval()
            testing_all_loss = 0.0
            testing_CE_loss = 0.0
            testing_sequence_loss = 0.0
            acc = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(testset):
                    data = data.cuda()
                    target = target.long().cuda()
                    output = model(data)['out']
                    
                    dg_index = (target > 0)
                    ce_loss = loss_func_1(output, target)[dg_index].mean()
                    sequence_loss = compute_sequence_loss(output, target)
                    loss = ce_loss + sequence_loss

                    testing_sequence_loss += sequence_loss.item()
                    testing_CE_loss += ce_loss.item()
                    testing_all_loss += loss.item()
                    acc += get_acc(output,target, dg_index)
                tqdm.write(f"[*]eval on test finish, loss is: {testing_all_loss/(i+1):.8f}, CE loss: {testing_CE_loss/(i+1):.8f}, seq loss: {testing_sequence_loss/(i+1):.8f}, acc is {100*acc/(i+1):.3f}%")
                writer.add_scalar('Eval on Test All Loss', testing_all_loss / (i+1), epoch)
                writer.add_scalar('Eval on Test CE Loss', testing_CE_loss / (i+1), epoch)
                writer.add_scalar('Eval on Test Sequence Loss', testing_sequence_loss / (i+1), epoch)
                writer.add_scalar('Eval on Test Acc', acc/(i+1), epoch)
                checkpoint = {
                "net": model.state_dict(),
                "epoch": epoch,
                }
                torch.save(checkpoint,save_path+save_name+"_"+str(epoch)+".pth")