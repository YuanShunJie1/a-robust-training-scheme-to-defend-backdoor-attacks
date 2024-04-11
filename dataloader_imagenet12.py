import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import torch.nn.functional as F

from torchvision.transforms import functional as Ft

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from autoaugment import CIFAR10Policy, ImageNetPolicy


# def ImageNetDataset(root, batch_size=256, workers=5, pin_memory=True):
traindir = './ImageNet12/imagenet12/train'
valdir = './ImageNet12/imagenet12/val'


MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET  = (0.229, 0.224, 0.225)  
        
tf_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])

tf_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])

tf_train_strong_10 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])

train_dataset = datasets.ImageFolder(traindir)
val_dataset = datasets.ImageFolder(valdir)


class imagenet_dataset(Dataset): 
    def __init__(self, dataset, mode, pred=[], probability=[]): 
        # self.transform = transform
        self.mode = mode
        
        if self.mode == "labeled":
            pred_idx = (1-pred).nonzero()[0]
            
        elif self.mode == "unlabeled":
            pred_idx = pred.nonzero()[0]
    
        self.train_data = [dataset.dataset[i] for i in pred_idx]
        
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target = self.train_data[index][0], self.train_data[index][1]
            img = Image.fromarray(img)
            img1 = tf_train(img) 
            img2 = tf_train(img)
            img3 = tf_train_strong_10(img)
            img4 = tf_train_strong_10(img)
            return img1, img2, img3, img4, target            
        
        elif self.mode=='unlabeled':
            img, target = self.train_data[index][0], self.train_data[index][1]
            img = Image.fromarray(img)
            img1 = tf_train(img) 
            img2 = tf_train(img)
            img3 = tf_train_strong_10(img)
            img4 = tf_train_strong_10(img)
            
            return img1, img2, img3, img4,target
           
    def __len__(self):
        return len(self.train_data)
     

def get_labeled_loader(args, dataset, pred=[], probability=[]):
    dataset = imagenet_dataset(dataset=dataset, mode='labeled', pred=pred, probability=probability)
    trainloader = DataLoader(dataset=dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return dataset, trainloader

def get_unlabeled_loader(args, dataset, pred=[], probability=[]):
    dataset = imagenet_dataset(dataset=dataset, mode='unlabeled', pred=pred, probability=probability)
    trainloader = DataLoader(dataset=dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return dataset, trainloader


def get_backdoor_loader(args):
    print('==> Preparing train data..')
    trainset = train_dataset

    train_data_bad = DatasetBD(args, full_dataset=trainset, inject_portion=args.inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,batch_size=args.batch_size*2, shuffle=False)

    return train_data_bad, train_bad_loader


def get_test_loader(args):
    print('==> Preparing test data..')
    testset = val_dataset

    test_data_clean = DatasetBD(args, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(args, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean, batch_size=args.batch_size*2, shuffle=False)
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad, batch_size=args.batch_size*2, shuffle=False)

    return test_clean_loader, test_bad_loader



class DatasetBD(Dataset):
    def __init__(self, args, full_dataset, inject_portion, transform=None, mode="train", distance=1):
        self.args = args
        self.dataset, self.poison_indices = self.addTrigger(full_dataset, args.target_label, inject_portion, mode, distance, args.trig_w, args.trig_h, args.trigger_type, args.target_type)
        self.device = torch.device("cuda:" + str(args.gpuid))
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label, index
    
    def __sort_dataset__(self, indices):
        box = [self.dataset[i] for i in indices]
        self.dataset = box
        
    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + " bad Imgs")

        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_, perm


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, mode, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'CLTrigger', 'dynamicTrigger', 'nashvilleTrigger',
                               'onePixelTrigger', 'wanetTrigger', 'blendTrigger']

        to224 = transforms.Compose([
                transforms.ToPILImage(),    
                transforms.Resize(256),
                transforms.CenterCrop(224)])
        img = to224(img)
        img = np.array(img)

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img)
        
        else:
            raise NotImplementedError

        return img

    def _blendTrigger(self, img, pattern=None, weight=None):
        width, height, c = img.shape
        
        if pattern is None:
            pattern = torch.zeros((1, width, height), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255
        else:
            pattern = pattern
            if pattern.dim() == 2:
                pattern = pattern.unsqueeze(0)

        if weight is None:
            weight = torch.zeros((1, width, height), dtype=torch.float32)
            weight[0, -3:, -3:] = 1.0
        else:
            weight = weight
            if weight.dim() == 2:
                weight = weight.unsqueeze(0)

        # Accelerated calculation
        res = weight * pattern
        weight = 1.0 - weight
        
        # transforms.ToPILImage()
        img = Image.fromarray(img)
        # print(type(img))
        img = Ft.pil_to_tensor(img)
        img = (weight * img + res).type(torch.uint8)
        img = img.permute(1, 2, 0).numpy()

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('/home/shunjie/experinment/robust_training_against_backdoor/ours/DivideMix-master/trigger/imagenet_sig_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('/home/shunjie/experinment/robust_training_against_backdoor/ours/DivideMix-master/trigger/ImageNet-trojan-mask.npy')
        # trg.shape: (3, 32, 32)
        # trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

