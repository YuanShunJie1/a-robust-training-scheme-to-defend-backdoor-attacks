from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy, ImageNetPolicy

MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)  
        
tf_train = transforms.Compose([
           transforms.ToPILImage(),
           transforms.RandomCrop(32, padding=4),
           # transforms.RandomRotation(3),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])

tf_test = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])

tf_train_strong_10 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])


# def unpickle(file):
#     import _pickle as cPickle
#     with open(file, 'rb') as fo:
#         dict = cPickle.load(fo, encoding='latin1')
#     return dict


class cifar_dataset(Dataset): 
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
            
            img1 = tf_train(img) 
            img2 = tf_train(img)
            img3 = tf_train_strong_10(img)
            img4 = tf_train_strong_10(img)
            return img1, img2, img3, img4, target            
        
        elif self.mode=='unlabeled':
            img, target = self.train_data[index][0], self.train_data[index][1]

            img1 = tf_train(img) 
            img2 = tf_train(img)
            img3 = tf_train_strong_10(img)
            img4 = tf_train_strong_10(img)
            
            return img1, img2, img3, img4,target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         

def get_labeled_loader(args, dataset, pred=[], probability=[]):
    dataset = cifar_dataset(dataset=dataset, mode='labeled', pred=pred, probability=probability)
    trainloader = DataLoader(dataset=dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return dataset, trainloader

def get_unlabeled_loader(args, dataset, pred=[], probability=[]):
    dataset = cifar_dataset(dataset=dataset, mode='unlabeled', pred=pred, probability=probability)
    trainloader = DataLoader(dataset=dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return dataset, trainloader
