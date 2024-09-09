import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import sys
import argparse
import os
import time
from torch.nn.parallel import DataParallel
from collections import OrderedDict 
SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from dataset.cifar10 import get_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl import *
from utils_ssl import loss_soft_reg_ep

from ssl_networks import CNN as MT_Net
from PreResNet import PreactResNet18_WNdrop
from wideArchitectures import WRN28_5_wn
import models_teacher.wideresnet as wrn_models
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=500, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='Training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dataset_type', default='ssl', help='How to prepare the data: only labeled data for the warmUp ("ssl_warmUp") or unlabeled and labeled for the SSL training ("ssl")')
    parser.add_argument('--train_root', default='./data', help='Root for train data')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='Number of labeled samples')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--download', type=bool, default=True, help='Download dataset')
    parser.add_argument('--network', type=str, default= 'MT_Net', help='The backbone of the network')  #### 'WRN28_5_wn' , 'MT_Net'
    parser.add_argument('--seed', type=int, default=47, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='Name of the experiment (for the output files)')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='The loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--num_classes', type=int, default=10, help='Beta parameter for the EMA in the soft labels')
    parser.add_argument('--dropout', type=float, default=0.0, help='CNN dropout')
    parser.add_argument('--load_epoch', type=int, default=0, help='Load model from the last epoch from the warmup')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='Alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='Set to 1 to choose the second gpu')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--swa', type=str, default='False', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=75, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.001, help='LR')
    parser.add_argument('--labeled_batch_size', default=250, type=int, metavar='N', help="Labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--val_samples', type=int, default=0, help='Number of samples to be kept for validation (from the training set))')
    parser.add_argument('--DA', type=str, default='standard', help='Chose the type of DA')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test):

    if args.validation_exp == "False":
        args.val_samples = 0

    ####################################### Train ##########################################################
    trainset, unlabeled_indexes, labeled_indexes, valset = get_dataset(args, transform_train, transform_test)

    if args.labeled_batch_size > 0 and not args.dataset_type == 'ssl_warmUp':
        print("Training with two samplers. {0} clean samples per batch".format(args.labeled_batch_size))
        batch_sampler = TwoStreamBatchSampler(unlabeled_indexes, labeled_indexes, args.batch_size, args.labeled_batch_size)
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers= 12, pin_memory=True)
    else:
        print(len(trainset))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers= 12, pin_memory=True)
        print('train loader is ssl_warmup')

    if args.validation_exp == "True":
        print("Training to choose hyperparameters --- VALIDATON MODE ---.")
        testset = valset
    else:
        print("Training to compare to the SOTA --- TESTING MODE ---.")
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        print("len of test set is ",len(testset))

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # train and val
    print('-------> Data loading')
    print("Training with {0} labeled samples ({1} unlabeled samples)".format(len(labeled_indexes), len(unlabeled_indexes)))
    return train_loader, test_loader, unlabeled_indexes, labeled_indexes


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def create_teacher_model():
        
    print("==> creating WideResNet" + str(28) + '-' + str(5))
    model = wrn_models.WideResNet(first_stride =  1,
                                            num_classes  = 10,
                                            depth        = 28,
                                            widen_factor = 5,
                                            activation   = 'relu')

    return model
        

def main(args):
    best_ac = 0.0

    model_teacher = create_teacher_model()

    #####################
    # Initializing seeds and preparing GPU
    #if args.cuda_dev == 1:
    #    torch.cuda.set_device(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)
    #####################

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    if args.DA == "standard":
        transform_train = transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])

    elif args.DA == "jitter":
        transform_train = transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])
    else:
        print("Wrong value for --DA argument.")


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    # data lodaer
    train_loader, test_loader, unlabeled_indexes, labeled_indexes = data_config(args, transform_train, transform_test)

    print("len of unlabeled indexes is ", len(unlabeled_indexes))

    if args.network == "MT_Net":
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)
        # model = create_teacher_model()
        print('model and teacher model are same wide resnet 28 5 for this experiment')

    elif args.network == "WRN28_5_wn":
        print("Loading WRN28_2...")
        model = WRN28_5_wn(num_classes = args.num_classes, dropout = args.dropout).to(device)
        print('#############################################################')
        model_teacher = create_teacher_model()
        print('model and teacher model are same for this exp')

    elif args.network == "PreactResNet18_WNdrop":
        print("Loading preActResNet18_WNdrop...")
        model = PreactResNet18_WNdrop(drop_val = args.dropout, num_classes = args.num_classes).to(device)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print("using", torch.cuda.device_count(),"GPUs!")                  #### I added code here
        model = model.to(device)

    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    milestones = args.M

    if args.swa == 'True':
        # to install it:
        # pip3 install torchcontrib
        # git clone https://github.com/pytorch/contrib.git
        # cd contrib
        # sudo python3 setup.py install
        from torchcontrib.optim import SWA
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        swa_optimizer = SWA(optimizer, swa_lr=args.swa_lr, swa_start=args.swa_start, swa_freq=args.swa_freq)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)


    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5, verbose=True)

    loss_train_epoch = []
    Rob_acc = []
    acc_train_per_epoch = []
    Nat_acc = []

    exp_path = os.path.join('./', 'ssl_models_{0}'.format(args.experiment_name), str(args.labeled_samples), 'trades_attack_wrn_seed_{0}'.format(args.seed))
    res_path = os.path.join('./', 'metrics_{0}'.format(args.experiment_name), str(args.labeled_samples), 'trades_attack_wrn_seed_{0}'.format(args.seed))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    cont = 0
    load = False
    save = True

    if args.load_epoch != 0:
        load_epoch = args.load_epoch
        load = True
        save = False

    if args.dataset_type == 'ssl_warmUp':
        load = False
        save = True
        
    if load:
        if args.loss_term == 'Reg_ep':
            train_type = 'C'
        if args.loss_term == 'MixUp_ep':
            train_type = 'M'
        if args.dropout > 0.0:
            train_type = train_type + 'drop' + str(int(10*args.dropout))
        if args.reg2 == 0.0:
            train_type = train_type + 'noReg'

        directory_path = './checkpoints/trades/mixup/'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


        path = './checkpoints/trades/mixup/warmUp_{0}_{1}_{2}_{3}_{4}_{5}_S{6}.hdf5'.format(train_type, \
                                                                                args.Mixup_Alpha, \
                                                                                load_epoch, \
                                                                                args.dataset, \
                                                                                args.labeled_samples, \
                                                                                args.network, \
                                                                                args.seed)
        # path = 'checkpoint_paper/1000/last_epoch_50_RobAcc_0.40400_NatAcc_0.45690_labels_1000_bestValLoss_0.46370.pth'
        
        # path = 'ssl_models_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/opt_last_epoch_100_valLoss_26.42000_valAcc_0.53440_labels_4000_bestValLoss_0.53440.pth'       
        # path_teacher = "checkpoint_paper/best.pth.tar"       ### 4000 labels
        # path_teacher = "checkpoint_paper/500/last.pth.tar"     ### 500 labels
        # path_teacher = "checkpoint_paper/500/last.pth (3).tar"  ### 500 labels_new
        # path_teacher = "checkpoint_paper/1000/last.pth (4).tar"
        if args.labeled_samples == 4000:
            path_teacher = "checkpoint_paper/best.pth.tar"
        if args.labeled_samples == 500:
            path_teacher = "checkpoint_paper/500/last.pth (3).tar"
        if args.labeled_samples == 1000:
            path_teacher = "checkpoint_paper/1000/last.pth (4).tar"
            
        print(path_teacher)
        checkpoint_teacher = torch.load(path_teacher)
        model_teacher.load_state_dict(checkpoint_teacher['state_dict'])
        model_teacher = model_teacher.to(device)
        
        checkpoint = torch.load(path)
        
#         new_state_dict = OrderedDict()
#         for key, value in checkpoint['state_dict'].items():
#             new_key = "module." + key  # Add the prefix "module."
#             new_state_dict[new_key] = value

#         print("Path loaded: ", path)
        model.load_state_dict(checkpoint['state_dict'])
        
        print("Relabeling the unlabeled samples...")     
        model_teacher.eval()
        # model.eval()
        results = np.zeros((len(train_loader.dataset), 10), dtype=np.float32)
        for images, images_pslab, labels, soft_labels, index in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model_teacher(images)
            # outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            # prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)
            results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()
            # print(results)

        train_loader.dataset.update_labels(results, unlabeled_indexes)
        print("Start training...")

    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################    
    # if args.dataset_type == 'ssl_warmUp':
    #     writer = SummaryWriter("runs/mix_up/trades/trades_mixup_50k_511")
        

    writer = SummaryWriter("logs/tensorboard")

    prev_results = []  
    # Natural_acc_per_epoch = []
    # robust_acc_per_epoch = []

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        # train for one epoch
        print(args.experiment_name, args.labeled_samples)

        loss_per_epoch_train, \
        top_5_train_ac, \
        top1_train_ac, \
        train_time = train_CrossEntropy(args,model,model_teacher, device, \
                                        train_loader, optimizer, \
                                        epoch, unlabeled_indexes, prev_results)
        
        # prev_results.append(current_results)
        

        loss_train_epoch += [loss_per_epoch_train]

        # test
        if args.validation_exp == "True":
            loss_per_epoch_test, acc_val_per_epoch_i,  = validating(args, model, device, test_loader)
        else:
            # loss_per_epoch_test, acc_val_per_epoch_i = testing(args, model, device, test_loader, epsilon= 8/255, num_steps=10, step_size= (8/4)/255)
            # Natural_acc_per_epoch, robust_acc_per_epoch = testing_new(args, model, device, test_loader)
            Natural_acc_per_epoch, robust_acc_per_epoch = testing_new(args, model, device, test_loader)
            # Natural_acc_per_epoch.append(natural_acc)
            # robust_acc_per_epoch.append(robust_acc)

        # if args.dataset_type == 'ssl_warmUp':
        #     writer.add_scalar('trades_mixup_Loss/test', np.array(loss_per_epoch_test), epoch)
        #     writer.add_scalar('trades_mixup_accuracy/test', np.array(acc_val_per_epoch_i), epoch)
        
        
        # 
        scheduler.step(Natural_acc_per_epoch[-1])
        # scheduler.step()
                

        Rob_acc += robust_acc_per_epoch
        acc_train_per_epoch += [top1_train_ac]
        Nat_acc += Natural_acc_per_epoch
        # AA_acc += Auto_attack_acc_per_epoch
        
        # current_lr = get_last_lr(optimizer)
        # print(f'Epoch {epoch} - Learning Rate: {current_lr}')


        # writer.add_scalar('Loss/test', torch.tensor(loss_val_epoch[-1]), epoch)
        # writer.add_scalar('accuracy/test', torch.tensor(acc_val_per_epoch[-1]), epoch)


        ####################################################################################################
        #############################               SAVING MODELS                ###########################
        ####################################################################################################
        checkpoints_dir = './checkpoints/trades/mixup'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)

        if epoch == 1:
            best_acc_val = Natural_acc_per_epoch[-1]
            snapBest = 'best_epoch_%d_RobAcc_%.5f_NatAcc_%.5f_labels_%d_bestAccVal_%.5f' % (
                epoch, robust_acc_per_epoch[-1], Natural_acc_per_epoch[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if Natural_acc_per_epoch[-1] > best_acc_val:
                best_acc_val = Natural_acc_per_epoch[-1]

                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_RobAcc_%.5f_NatAcc_%.5f_labels_%d_bestAccVal_%.5f' % (
                    epoch, robust_acc_per_epoch[-1], Natural_acc_per_epoch[-1], args.labeled_samples, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1
        
        if epoch > 80:
            last_20_epochs_dir = os.path.join(exp_path, f'last_20_epochs_seed_{args.seed}_labels_{args.labeled_samples}')
            os.makedirs(last_20_epochs_dir, exist_ok=True)
            snapLast = 'epoch_%d_RobAcc_%.5f_NatAcc_%.5f_labels_%d_bestValLoss_%.5f' % (
                epoch, robust_acc_per_epoch[-1], Natural_acc_per_epoch[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(last_20_epochs_dir, snapLast + '.pth'))   
            
        if epoch == args.epoch:
            snapLast = 'last_epoch_%d_RobAcc_%.5f_NatAcc_%.5f_labels_%d_bestValLoss_%.5f' % (
                epoch, robust_acc_per_epoch[-1], Natural_acc_per_epoch[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

        ### Saving model to load it again
        # cond = epoch%1 == 0
        if args.dataset_type == 'ssl_warmUp':
            if args.loss_term == 'Reg_ep':
                train_type = 'C'
            if args.loss_term == 'MixUp_ep':
                train_type = 'M'
            if args.dropout > 0.0:
                train_type = train_type + 'drop' + str(int(10*args.dropout))
            if args.reg2 == 0.0:
                train_type = train_type + 'noReg'


            cond = (epoch==args.epoch)
            name = 'warmUp_{1}_{0}'.format(args.Mixup_Alpha, train_type)
            save = True
        else:
            cond = False


        if cond and save:
            print("Saving models...")
            path = './checkpoints/trades/mixup/{0}_{1}_{2}_{3}_{4}_S{5}.hdf5'.format(name, epoch, args.dataset, \
                                                                        args.labeled_samples, \
                                                                        args.network, \
                                                                        args.seed)

            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss_train_epoch' : np.asarray(loss_train_epoch),
                    'acc_robust' : np.asarray(Rob_acc),
                    'acc_train_per_epoch' : np.asarray(acc_train_per_epoch),
                    'acc_natural' : np.asarray(Nat_acc),
                    'labels': np.asarray(train_loader.dataset.soft_labels)
                }, filename = path)



        ####################################################################################################
        ############################               SAVING METRICS                ###########################
        ####################################################################################################



        # Save losses:
        np.save(f"{res_path}/{args.labeled_samples}_LOSS_epoch_train_epoch_{args.epoch}.npy", np.asarray(loss_train_epoch))
        # np.save(res_path + '/' + str(args.labeled_samples) + '_LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # save accuracies:
        np.save(f"{res_path}/{args.labeled_samples}_Robust_accuracy_epoch_{args.epoch}.npy", np.asarray(Rob_acc))
        np.save(f"{res_path}/{args.labeled_samples}_Natural_accuracy_epoch_{args.epoch}.npy", np.asarray(Nat_acc))
        # np.save(f"{res_path}/{args.labeled_samples}_Auto-Attack_accuracy_epoch_{args.epoch}.npy", np.asarray(AA_acc))

    # applying swa
#     if args.swa == 'True':
#         optimizer.swap_swa_sgd()
#         optimizer.bn_update(train_loader, model, device)
#         if args.validation_exp == "True":
#             loss_swa, acc_val_swa = validating(args, model, device, test_loader)
#         else:
#             loss_swa, acc_val_swa = testing(args, model, device, test_loader)

#         snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_labels_%d_bestValLoss_%.5f_swaAcc_%.5f' % (
#             epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val, acc_val_swa[0])
#         torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
#         torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

    print('Best ac:%f' % best_acc_val)



if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)