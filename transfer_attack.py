from __future__ import print_function
import os
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from models.wide_resnet import WRN28_10
from models.resnet import RN18_10
from models.denoise import Denoise,Conv
from dataset.cifar10 import CIFAR10
from utils import *
from adv_generator import *

parser = argparse.ArgumentParser(description='PyTorch DAD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoint/CIFAR10/Denoise',
                    help='directory of model for saving checkpoint')
parser.add_argument('--mmd-dir', default='./checkpoint/CIFAR10/SAMMD',
                    help='directory of mmd for saving checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='wrn28', choices=['rn18', 'wrn70', 'rn50', 'swin'])
parser.add_argument("--mmd-batch", type=int, default=100, help="batch size for mmd training")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--attack',type=str,default='adaptive_pgd',help='select attack setting')
parser.add_argument('--mode', type=str, default='test', help='decide to generate test data or train data')
parser.add_argument('--epsilon', default=12/255, type=parse_fraction, help='perturbation')
parser.add_argument('--threshold', default=0.02, type=float, help='threshold for mmd')
parser.add_argument('--num-steps', default=10, type=int, help='perturb number of steps')
parser.add_argument('--index', type=int, default=1, help='index of the model')
parser.add_argument('--white-box', action='store_true', default=False, help='white-box attack or non-white-box attack')
args = parser.parse_args()

def eval_test(denoiser, clf, device, test_loader, nat_data, semantic_model, sigma, sigma0, ep):
    denoiser.eval()
    clf.eval()
    correct = 0
    
    std = 0.25
    mean = 0

    for data, label in test_loader:
        if nat_data.shape[0] > data.shape[0]:
            nat_data = nat_data[0:data.shape[0]]
            
        data, label = data.to(device), label.to(device)

        nat_feature = semantic_model(nat_data)
        nat_feature = nat_feature.view(nat_feature.size(0), -1)

        test_feature = semantic_model(data)
        test_feature = test_feature.view(test_feature.size(0), -1)

        S = torch.cat([nat_feature.cpu(), test_feature.cpu()], 0).cuda()
        Sv = S.view(nat_feature.shape[0] + test_feature.shape[0], -1)
        
        S = torch.cat([nat_data.cpu(), data.cpu()], 0).cuda()
        S_FEA = S.view(nat_data.shape[0] + test_feature.shape[0], -1)

        _, _, mmd_value= SAMMD_WB(Sv, 100, nat_feature.shape[0], S_FEA, 
                                   sigma, sigma0, ep, 0.05, device, torch.float)

        #add Gaussian noise
        noise = torch.randn(data.size()) * std + mean
        noise = noise.to(device)
        noise = torch.clamp(noise, 0, 1)
        noisy_data = torch.clamp(data + noise, 0, 1)

        if mmd_value < args.threshold:
            logits_out = clf(data)
        else:
            X_puri = denoiser(noisy_data)
            logits_out = clf(X_puri)
        
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

    print('Test Accuracy: {}/{} ({:.2f}%)'.format(correct,
        len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def main():
    # settings for CIFAR10
    setup_seed(1)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu") 

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()

    if args.model == 'rn18':
        adv_dir = './adv_data/CIFAR10/RN18'
    elif args.model == 'wrn70':
        adv_dir = './adv_data/CIFAR10/WRN70'
    elif args.model == 'rn50':
        adv_dir = './adv_data/CIFAR10/RN50'
    elif args.model == 'swin':
        adv_dir = './adv_data/CIFAR10/SWIN'

    if args.epsilon == 8/255:
        cw_epsilon = 0.5
    elif args.epsilon == 12/255:
        cw_epsilon = 1.0

    eotpgd_testset = torch.load('{}/test_{}_eotpgd_{}_{}_{}.pth'.format(adv_dir, args.white_box, args.epsilon, args.model, args.index))
    eotpgd_test_loader = DataLoader(eotpgd_testset, batch_size=args.batch_size, shuffle=False)

    cw_testset = torch.load('{}/test_{}_cw_{}_{}_{}.pth'.format(adv_dir, args.white_box, cw_epsilon, args.model, args.index))
    cw_test_loader = DataLoader(cw_testset, batch_size=args.batch_size, shuffle=False)

    denoiser_dir = './checkpoint/CIFAR10/Denoise'
    input_size = [32, 32]
    block = Conv
    fwd_out = [64, 128, 256, 256, 256]
    num_fwd = [2, 3, 3, 3, 3]
    back_out = [64, 128, 256, 256]
    num_back = [2, 3, 3, 3]
    denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
    denoiser = torch.nn.DataParallel(denoiser)

    semantic_model = RN18_10(semantic=True).to(device)
    semantic_model = torch.nn.DataParallel(semantic_model)
    semantic_checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
    semantic_model.load_state_dict(semantic_checkpoint)
    semantic_model.eval()
    print('load semantic model successfully!')

    clf = WRN28_10(semantic=False).to(device)
    clf = torch.nn.DataParallel(clf)
    checkpoint = torch.load('checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth')
    clf.load_state_dict(checkpoint)
    clf.eval()
    print('load cls successfully!')

    train_dataset = train_loader.dataset

    # create subsets
    with open('data/cifar10_mmd_indices.pkl', 'rb') as f:
        mmd_indices = pickle.load(f)
    print('load mmd indices successfully!')
    args.mmd_batch = len(mmd_indices)
    mmd_subset = Subset(train_dataset, mmd_indices)
    print('load mmd dataset successfully!')
    data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
    nat_data = torch.stack(data_only)

    cudnn.benchmark = True

    denoiser_ckpt = torch.load('{}/CIFAR10_wrn28_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.index))
    denoiser.load_state_dict(denoiser_ckpt)

    loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir))
    ep = loaded_parameters['ep']
    sigma0 = loaded_parameters['sigma0']
    sigma = loaded_parameters['sigma']

    print("Current model is: ", args.model)
    print('====================EOTPGD Results===========================')
    eval_test(denoiser, clf, device, eotpgd_test_loader, nat_data, semantic_model, sigma, sigma0, ep)
    print('====================CW Results===========================')
    eval_test(denoiser, clf, device, cw_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

if __name__ == '__main__':
    main()