import torchvision as tv
import numpy as np
from PIL import Image
import time
import sys
sys.path.append('../utils_pseudoLab/')
from scipy.special import softmax
import torch
import torch.nn.functional as F
from wideArchitectures import WRN28_5_wn
import models_teacher.wideresnet as wrn_models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def create_teacher_model():        
    print("==> creating WideResNet" + str(28) + '-' + str(5))
    model = wrn_models.WideResNet(first_stride =  1,
                                            num_classes  = 10,
                                            depth        = 28,
                                            widen_factor = 5,
                                            activation   = 'relu')

    return model
        

def get_dataset(args, transform_train, transform_val):
    # prepare datasets
    cifar10_train_val = tv.datasets.CIFAR10(args.train_root, train=True, download=args.download)

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(args, cifar10_train_val.targets)   ##changed tagets to train_labels
    print("len of train data is", len(train_indexes))
    print("len of val data is", len(val_indexes))
    train = Cifar10Train(args, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
    print("len is", len(train.data))

    validation = Cifar10Train(args, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'ssl_warmUp':
        #print('ssl_data only labeled data')
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl_warmUp()
    elif args.dataset_type == 'ssl':
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl()

    return train, unlabeled_indexes, labeled_indexes, validation


def train_val_split(args, train_val):

    np.random.seed(args.seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int( args.val_samples / args.num_classes)
    #print("val_num is ", val_num)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    print("len of train indexes is " ,len(train_indexes))
    print("len of val indexes is " ,len(val_indexes))
    
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar10Train(tv.datasets.CIFAR10):
    def __init__(self, args, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar10Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform,                   download=download)
        self.args = args
        if train_indexes is not None:
            self.data = self.data[train_indexes]
            self.targets = np.array(self.targets)[train_indexes]
            print('length of self.targets $$$$####', len(self.targets))
        self.soft_labels = np.zeros((len(self.targets), 10), dtype=np.float32)
        self._num = int(len(self.targets) - int(args.labeled_samples))
        #print("self._num is",self._num)
        print("len of train_labels is ", int(len(self.targets)))
        print("len of args.labeled_samples",int(args.labeled_samples))
        self.original_labels = np.copy(self.targets)
        self.pslab_transform = pslab_transform
        
        
    def prepare_data_ssl(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)
        unlabeled_indexes = [] # initialize the vector
        labeled_indexes = []
        
        file_paths = {
        500: 'checkpoint_paper/500/sampled_label_idx_500 (4).npy',
        1000: 'checkpoint_paper/1000/sampled_label_idx_1000.npy',
        4000: 'checkpoint_paper/sampled_label_idx_4000.npy'}
        
        # labeled_indexes_from_numpy = np.load('checkpoint_paper/sampled_label_idx_4000.npy')  ### 4000 labels
        # labeled_indexes_from_numpy = np.load('checkpoint_paper/500/sampled_label_idx_500 (1).npy')    ### 500 labels_old
        # labeled_indexes_from_numpy = np.load('checkpoint_paper/500/sampled_label_idx_500 (4).npy')## 500 _new
        # labeled_indexes_from_numpy = np.load('checkpoint_paper/1000/sampled_label_idx_1000.npy')
        labeled_indexes_path = file_paths[self.args.labeled_samples]
        labeled_indexes_from_numpy = np.load(labeled_indexes_path)

        num_unlab_samples = self._num
        num_labeled_samples = len(self.targets) - num_unlab_samples
        
        labeled_indexes = labeled_indexes_from_numpy
        all_indexes = set(range(len(self.targets)))
        unlabeled_indexes = list(all_indexes - set(labeled_indexes_from_numpy))
        print("length of labelled samples are", len(labeled_indexes))
        print("length of unlabelled samples are", len(unlabeled_indexes))
        
        for i, label in enumerate(self.targets):
            self.soft_labels[i][label] = 1.0
        
        return np.asarray(unlabeled_indexes),  np.asarray(labeled_indexes)   
    
    
    
    def prepare_data_ssl_original(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)
        unlabeled_indexes = [] # initialize the vector
        labeled_indexes = []


        num_unlab_samples = self._num
        num_labeled_samples = len(self.targets) - num_unlab_samples

        labeled_per_class = int(num_labeled_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            for i in range(len(indexes)):
                if i < unlab_per_class:
                    label_sym = np.random.randint(self.args.num_classes, dtype=np.int32)
                    self.targets[indexes[i]] = label_sym

                self.soft_labels[indexes[i]][self.targets[indexes[i]]] = 1

            unlabeled_indexes.extend(indexes[:unlab_per_class])
            labeled_indexes.extend(indexes[unlab_per_class:])

        return np.asarray(unlabeled_indexes),  np.asarray(labeled_indexes)
    
    
    def prepare_data_ssl_warmUp_50000(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)
        unlabeled_indexes = [] # initialize the vector
        train_indexes = []

        num_unlab_samples = self._num
        num_labeled_samples = len(self.targets) - num_unlab_samples

        labeled_per_class = int(num_labeled_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            unlabeled_indexes.extend(indexes[:unlab_per_class])
            train_indexes.extend(indexes[unlab_per_class:])

        np.asarray(train_indexes)
        print('len of train_indexes in prepare_Data_Ssl$$$$$$$',len(train_indexes))

        self.data = self.data[train_indexes]
        self.targets = np.array(self.targets)[train_indexes]
        self.soft_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.data)):
            self.soft_labels[i][self.targets[i]] = 1

        # for i in (train_indexes):
        #     self.soft_labels[i][self.train_labels[i]] = 1

        unlabeled_indexes = np.asarray([])

        return np.asarray(unlabeled_indexes), np.asarray(train_indexes)
    
    
    
    def prepare_data_ssl_warmUp(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)
        unlabeled_indexes = []  # initialize the vector
        
        
        file_paths = {
        500: 'checkpoint_paper/500/sampled_label_idx_500 (4).npy',
        1000: 'checkpoint_paper/1000/sampled_label_idx_1000.npy',
        4000: 'checkpoint_paper/sampled_label_idx_4000.npy' }

        # # Load labeled indexes from the numpy file
        # # labeled_indexes = np.load('checkpoint_paper/sampled_label_idx_4000.npy') ## 4000 labels
        # labeled_indexes = np.load('checkpoint_paper/500/sampled_label_idx_500 (4).npy')    ### 500 labels
        # # labeled_indexes = np.load('checkpoint_paper/1000/sampled_label_idx_1000.npy')
        labeled_indexes_path = file_paths[self.args.labeled_samples]
        labeled_indexes = np.load(labeled_indexes_path)

        train_indexes = labeled_indexes.tolist()

        # Get the total number of samples
        total_samples = len(self.targets)
    
        # Determine the unlabeled indexes by excluding labeled indexes
        all_indexes = set(range(total_samples))
        labeled_indexes_set = set(labeled_indexes)
        unlabeled_indexes = list(all_indexes - labeled_indexes_set)

        # Shuffle unlabeled indexes
        np.random.shuffle(unlabeled_indexes)

        np.asarray(train_indexes)
        print('len of train_indexes in prepare_Data_Ssl::::', len(train_indexes))

        self.data = self.data[train_indexes]
        self.targets = np.array(self.targets)[train_indexes]
        print('length of self.targets is%%%%%%', len(self.targets))
        self.soft_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.data)):
            self.soft_labels[i][self.targets[i]] = 1
        
        unlabeled_indexes = np.asarray([])    

        return np.asarray(unlabeled_indexes), np.asarray(train_indexes)
    
    
        

    def update_labels(self, result, unlabeled_indexes):
        relabel_indexes = list(unlabeled_indexes)

        self.soft_labels[relabel_indexes] = result[relabel_indexes]
        self.targets[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)

        print("Samples relabeled with the prediction: ", str(len(relabel_indexes))) 




    def __getitem__(self, index):
        #print(index)
        img, labels, soft_labels = self.data[index], self.targets[index], self.soft_labels[index]
        img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, img_pseudolabels, labels, soft_labels, index
