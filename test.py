import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as trans
import argparse
import torchattacks
import torch.distributed as dist
from dataset import Dataset
import torchattacks
import random
from wideresnet import WideResNet
from resnets import *
from PGD import PGD
from CW_inf_attack import CW_linf


def test(model, test_data, args, mode='test'):
    if args.attack != None and mode == 'test':
        if args.attack == 'AutoAttack':
            attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        elif args.attack == 'APGD':
            attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        elif args.attack == 'PGD':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
        elif args.attack == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        elif args.attack == 'CW':
            attack = CW_linf(model)
        elif args.attack == 'Square':
            attack = torchattacks.Square(model, norm='Linf', eps=8/255)
            
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False)
    correct = 0
    correct_adv = 0
    all_samples = 0
    device = next(model.parameters()).device
    model = model.eval()
    batch_id = 0
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465])
    cifar10_std = torch.tensor([0.2471, 0.2435, 0.2616])
    # epoch = args.epoch
    # data_norm = trans.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    with tqdm(test_loader) as test_loader:
        for data, label in test_loader:
            batch_id+=1
            data, label = data.to(device), label.to(device)

            outputs = model(data)
            correct += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            if args.attack != 'None' and mode == 'test':
                data = attack(data, label)
                outputs = model(data)
                correct_adv += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            all_samples += data.shape[0]
            if args.attack == 'AutoAttack':
                print('batch_id: {} \t Attack: {} \t Accuracy = {}/{} = {:.4f}'.format(batch_id,args.attack, correct_adv, all_samples, correct_adv*1.0 / all_samples))
                # if batch_id == 1:
                #     break

    acc_ori = correct*1.0 / all_samples
    print('Attack: None \t Accuracy = {}/{} = {:.4f}'.format(correct, all_samples, acc_ori))
    acc_adv = 0
    if args.attack != 'None' and mode == 'test':
        acc_adv = correct_adv*1.0 / all_samples
        print('Attack: {} \t Accuracy = {}/{} = {:.4f}'.format(args.attack, correct_adv, all_samples, acc_adv))
        print(acc_ori - acc_adv)
    if mode =='train':
        return acc_ori  
    else:
        return acc_ori, acc_adv


# python test.py --weights ./checkpoint/best-ResNet18--CIFAR10-PGD.pt --attack PGD --step 50 --dataset CIFAR10 --model resnet18
"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            # print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)
    return net


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--weights', type=str, default='./checkpoint/BORT-WRN28-10-CIFAR10.pt', help='saved model path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    parser.add_argument('--attack', type=str, default='None', help='attack method')
    parser.add_argument('--step', type=int, default=10, help='attack method')
    parser.add_argument('--model', type=str, default='resnet18', help='tested model')
    parser.add_argument('--total_scales', type=int, default=5, help='multi-filter scaling parameter') 
    parser.add_argument('--ButterworthKS', type=int, default=7, help='multi-filter scaling parameter') 
    args = parser.parse_args()
    device = 'cuda'

    num_classes = 10
    if args.dataset == 'CIFAR100':
        num_classes = 100
    
    assert args.model in ['resnet18', 'wrn-34-10','resnet18_dnet',"resnet18_tfnet","resnet18_mnet","resnet18_bnet"]
    if args.model == 'resnet18':
        model = ResNet18().cuda()
    elif args.model == 'resnet18_dnet':
        model = ResNet18_DNet(num_classes).cuda()
    elif args.model == 'resnet18_tfnet':
        model = ResNet18_TFNet(num_classes).cuda()
    elif args.model == 'resnet18_mnet':
        model = ResNet18_MNet(num_classes,args.total_scales).cuda()
    elif args.model == 'resnet18_bnet':
        model = ResNet18_BNet(num_classes, args.ButterworthKS).cuda()
    else:
        model = WideResNet(num_classes=num_classes).cuda()

    # args.weights = f"./checkpoint/CIFAR10-{args.model}-{args.ButterworthKS}.pt"

    # weights = torch.load(args.weights)
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict, strict=False)
    model = torch.load(args.weights)
    model = model.to(device)
    print(model)
    test_data = Dataset(path = args.data_root, dataset = args.dataset, train = False)


    acc = test(model, test_data, args)
