from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from models.vgg_cifar import vgg16
from models.mobilenetv2 import MobileNetV2
from sklearn.mixture import GaussianMixture
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from poison_tool_cifar import get_backdoor_loader, get_test_loader
from dataloader_cifar import get_labeled_loader, get_unlabeled_loader
import ssl_tool as st


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--id', default=1, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./CIFAR10/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--resume_path', default='./checkpoint', type=str, help='path to dataset')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_workers', default=5, type=int)
parser.add_argument('--model_type', default='resnet18', type=str)
parser.add_argument('--weight_decay', default=2, type=float, help='weight_decay')

# 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger', 'CLTrigger', 'dynamicTrigger', 'nashvilleTrigger', 'onePixelTrigger', 'wanetTrigger'

# backdoor attacks
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor poisoned data')

# SSL
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda_u', default=15, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--rampup_length', default=190, type=int)
parser.add_argument('--train_iteration', default=1024, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)


stats_log=open(f'{args.resume_path}/stats_network={args.model_type}_dataset={args.dataset}_id={args.id}_trig_type={args.trigger_type}_target_label={args.target_label}_target_type={args.target_type}_trig_w={args.trig_w}_trig_h={args.trig_h}_pr={args.inject_portion}.txt','w') 
test_log=open(f'{args.resume_path}/test_network={args.model_type}_dataset={args.dataset}_id={args.id}_trig_type={args.trigger_type}_target_label={args.target_label}_target_type={args.target_type}_trig_w={args.trig_w}_trig_h={args.trig_h}_pr={args.inject_portion}.txt','w')     


train_data_bad, backdoor_data_loader = get_backdoor_loader(args)
clean_test_loader, bad_test_loader = get_test_loader(args)


def warmup(num_epoch,net,dataloader,wu_lr=0.002):
    print('Warmup Net\n','lr',wu_lr)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=wu_lr, momentum=0.9, weight_decay=args.weight_decay)

    all_loss = []
    
    for epoch in range(num_epoch):
        box = torch.zeros(50000)
        num_iter = (len(dataloader.dataset)//(args.batch_size*2))+1
        for batch_idx, (inputs, labels, index) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            outputs, _ = net(inputs)               
            losses = F.cross_entropy(outputs, labels, reduction='none')
            loss = torch.mean(losses)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            loss.backward()  
            optimizer.step() 
            
            for b in range(inputs.size(0)):
                box[index[b]]=losses[b].item()
      
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'%(args.dataset, args.inject_portion, args.trigger_type, epoch, num_epoch, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()

        box = (box-box.min())/(box.max()-box.min())  
        all_loss.append(box)    
        test(epoch,net,clean_test_loader,mode='Clean')
        test(epoch,net,bad_test_loader,mode='Poison')  

    return all_loss


def train_with_clean_data(num_epoch,net,dataloader,wu_lr=0.002):
    print('Train Net with Clean dataset\n')
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(num_epoch):
        if epoch >= 150:
            wu_lr /= 10      
        for param_group in optimizer.param_groups:
            param_group['lr'] = wu_lr  
        
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        for batch_idx, (inputs, _, _, _, labels) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            outputs, _ = net(inputs)               
            loss = F.cross_entropy(outputs, labels)      
            loss.backward()  
            optimizer.step() 
      
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'%(args.dataset, args.inject_portion, args.trigger_type, epoch, num_epoch, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()

        test(epoch,net,clean_test_loader,mode='Clean')
        test(epoch,net,bad_test_loader,mode='Poison')  
  

def test(epoch,net,test_loader,mode='Clean'):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    if mode == 'Clean':
        print("\n| Test Epoch #%d\t Clean Accuracy: %.2f%%" %(epoch,acc))
        test_log.write('Epoch:%d  Clean Accuracy:%.2f'%(epoch,acc))
        test_log.flush()  
    else:
        print("\n| Test Epoch #%d\t Poison Accuracy: %.2f%%\n" %(epoch,acc))
        test_log.write('Epoch:%d  Poison Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush()        


def acc_recall(pred, prob):
    ground_truth = np.zeros(len(pred))
    ground_truth[train_data_bad.poison_indices] = 1
    prediction = pred.astype(int)
    
    recall   = recall_score(ground_truth, prediction)
    accuracy = accuracy_score(ground_truth, prediction)
    
    print('Numer of poison samples:%d'%(pred.sum()))
    print('Recall:%.3f'%(recall))
    print('Accuracy:%.3f\n'%(accuracy))
    
    stats_log.write('Numer of poison samples:%d Recall:%.3f Accuracy:%.3f\n'%(pred.sum(), recall, accuracy))
    stats_log.flush()


def poison_detection(input_loss):
    print('Posion detection\n')
    
    input_loss = all_loss[-1]
    input_loss = input_loss.reshape(-1,1)

    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:,gmm.means_.argmin()]
    pred = prob > args.p_threshold
    
    return prob, pred    


def mixmatch(args, labeled_trainloader, unlabeled_trainloader, model, num_epochs, ssl_lr=0.002):
    train_criterion = st.SemiLoss()
    optimizer = optim.Adam(model.parameters(), lr=ssl_lr)
    start_epoch = 0

    lr = ssl_lr
    for epoch in range(start_epoch, num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr/10
        
        st.ssl_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch, num_epochs, use_cuda=True)
        
        test(epoch,net,clean_test_loader,mode='Clean')
        test(epoch,net,bad_test_loader,mode='Poison')  


def create_model():
    if args.model_type == 'resnet18':
        model = ResNet18(num_classes=args.num_class)
    if args.model_type == 'vgg16':
        model = vgg16(num_classes=args.num_class)   
    if args.model_type == 'MobileNetV2':
        model = MobileNetV2(num_classes=args.num_class)   

    model = model.cuda()
    return model


if __name__ == '__main__':
    net = create_model()
    cudnn.benchmark = True
    
    # Backdoor seperation
    all_loss = warmup(5, net, backdoor_data_loader)
    prob, pred = poison_detection(all_loss)
    acc_recall(pred, prob)
    clean_dataset, clean_trainloader = get_labeled_loader(args, train_data_bad, pred=pred, probability=prob)
    poison_dataset, poison_trainloader = get_unlabeled_loader(args, train_data_bad, pred=pred, probability=prob)
    print('Labeled data',len(clean_dataset), 'Unlabeled data',len(poison_dataset))
    
    # Supervisied training 
    net = create_model()
    train_with_clean_data(100,net,clean_trainloader)
    
    # Semi-supervisied fine-tuning
    mixmatch(args, clean_trainloader, poison_trainloader, net, 30, ssl_lr=0.002)


