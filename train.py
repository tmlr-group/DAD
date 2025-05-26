from __future__ import print_function
import os
import argparse
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from models.resnet import RN18_10
from models.wide_resnet import WRN28_10, WRN70_16
from torchvision.models import resnet50
from models.denoise import Denoise, Conv
from dataset.utils import *
from torchvision.datasets import CIFAR10 as DATA
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pickle
from utils import *
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch DAD')

parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--weight-decay', default=2e-4, type=float, metavar='W', help='weight decay (default: 2e-4)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument("--alpha", type=float, default=1e-2, help="regularization term")
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255, type=parse_fraction, help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--num-class', default=10, help='number of classes')
parser.add_argument('--step-size', default=2/255, type=parse_fraction, help='perturb step size')
parser.add_argument('--first-decay', type=int, default=45, help='adjust learning rate on which epoch in the first round')
parser.add_argument('--second-decay', type=int, default=60, help='adjust learning rate on which epoch in the second round')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/CIFAR10/Denoise', help='directory of model for saving checkpoint')
parser.add_argument('--mmd-dir', default='./checkpoint/CIFAR10/SAMMD', help='directory of mmd for saving checkpoint')
parser.add_argument('--adv-dir', default='./adv_data/CIFAR10/WRN28/', help='directory of adversarial data')
parser.add_argument('--save-freq', '-s', default=10, type=int, help='save frequency')
parser.add_argument('--test-freq', default=1, type=int, help='test frequency')
parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
parser.add_argument('--data', type=str, default='CIFAR10', help='data source', choices=['CIFAR10', 'ImageNet', 'SVHN'])
parser.add_argument('--model', type=str, default='wrn28', choices=['wrn28', 'wrn70', 'rn50', 'rn18'])
parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs for mmd training")
parser.add_argument("--mmd-batch", type=int, default=100, help="batch size for mmd training")
parser.add_argument('--mode', type=str, default='train', help='decide to train a denoiser or test the denoiser')
parser.add_argument('--attack', type=str, default='mma', help='select attack setting')
parser.add_argument('--index', type=int, default=1, help='index of the model')
parser.add_argument('--fpr', type=float, default=0.05, help='fpr for detecting AEs')
parser.add_argument('--type', type=str, default='vanilla', choices=['vanilla', 'trades', 'mart', 'awp'])
args = parser.parse_args()

def train_mmd(args, train_loader, ae_train_loader, semantic_model, device):

    for _, ((X_nat, label), (X_adv, _)) in enumerate(zip(train_loader, ae_train_loader)):
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, torch.float))
        epsilonOPT.requires_grad = True
        
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, torch.float)
        sigmaOPT.requires_grad = True

        X_nat, X_adv, label = X_nat.to(device), X_adv.to(device), label.to(device)

        nat_feature = semantic_model(X_nat)
        nat_feature = nat_feature.view(nat_feature.size(0), -1)
        adv_feature = semantic_model(X_adv)
        adv_feature = adv_feature.view(adv_feature.size(0), -1)

        # concatenate semantic features of natural and adversarial images
        S = torch.cat([nat_feature.cpu(), adv_feature.cpu()], 0).cuda()
        Sv = S.view(nat_feature.shape[0] + adv_feature.shape[0], -1)

        # concatenate natural and adversarial images
        S = torch.cat([X_nat.cpu(), X_adv.cpu()], 0).cuda()
        S_FEA = S.view(X_nat.shape[0] + X_adv.shape[0], -1)

        Dxy = Pdist2(Sv[:X_nat.shape[0], :], Sv[X_nat.shape[0]:, :])
            
        sigma0 = Dxy.median().detach()
        sigma0.requires_grad = True

        optimizer_sigma0 = torch.optim.Adam([sigma0]+[sigmaOPT]+[epsilonOPT], lr=0.0002)

        for t in range(args.n_epochs):
            ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            TEMPa = MMDu(Sv, args.mmd_batch, S_FEA, sigma, sigma0, ep, is_smooth=True)
            mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)

            optimizer_sigma0.step()
            if t % 100 == 0:
                print('printing training statistics...')
                print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                        -1 * STAT_adaptive.item())
                    
    return epsilonOPT.clone().detach(), sigma0.clone().detach(), sigmaOPT.clone().detach()

def train_denoiser(args, train_loader, ae_train_loader, optimizer, epoch, denoiser, semantic_model, 
                    clf, ep, sigma0, sigma, device):
    correct_adv = 0
    mean = 0.0
    std = 0.25
    
    for batch_idx, ((X_nat, label), (X_adv, _)) in enumerate(zip(train_loader, ae_train_loader)):
        denoiser.train()

        X_nat, X_adv, label = X_nat.to(device), X_adv.to(device), label.to(device)

        # add Gaussian noise to X_adv
        noise = torch.randn(X_adv.size()) * std + mean
        noise = noise.to(device)
        noise = torch.clamp(noise, 0, 1)
        X_adv = torch.clamp(X_adv + noise, 0, 1)

        clf.eval()

        optimizer.zero_grad()
        X_puri = denoiser(X_adv)
        
        nat_feature = semantic_model(X_nat)
        nat_feature = nat_feature.view(nat_feature.size(0), -1)
        puri_feature = semantic_model(X_puri)
        puri_feature = puri_feature.view(puri_feature.size(0), -1)
				
        S = torch.cat([nat_feature.cpu(), puri_feature.cpu()], 0).cuda()
        Sv = S.view(nat_feature.shape[0] + puri_feature.shape[0], -1)
        
        S = torch.cat([X_nat.cpu(), X_puri.cpu()], 0).cuda()
        S_FEA = S.view(X_nat.shape[0] + X_puri.shape[0], -1)
        
        _, _, mmd_value = SAMMD_WB(Sv, 100, X_nat.shape[0], S_FEA, 
                                   sigma, sigma0, ep, 0.05, device, torch.float)
        logits_out = clf(X_puri)
        loss = F.cross_entropy(logits_out, label)
        #total_loss = loss + 100*mmd_value
        total_loss = mmd_value + args.alpha * loss
        total_loss.backward()
        optimizer.step()

        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, (batch_idx+1) * len(X_nat), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader)))
            
    print('Robust Training Accuracy: {}/{} ({:.2f}%)'.format(
        correct_adv, len(train_loader.dataset), 100. * correct_adv / len(train_loader.dataset)))

def main():
    # settings
    setup_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # setup data loader
    if args.data == 'CIFAR10':
        train_dataset = DATA('./data/', train=True, download=True, 
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]))
        with open('data/cifar10_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')
        args.mmd_batch = len(mmd_indices)
        mmd_subset = Subset(train_dataset, mmd_indices)
        mmd_loader = DataLoader(mmd_subset, 
                          batch_size=args.mmd_batch, 
                          shuffle=False, 
                          num_workers=args.num_workers)
        print('load mmd dataset successfully!')
        with open('data/cifar10_train_indices.pkl', 'rb') as f:
            train_indices = pickle.load(f)
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          num_workers=args.num_workers)
        print('load train dataset successfully!')

        denoiser_dir = './checkpoint/CIFAR10/Denoise'
        input_size = [32, 32]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)

        if args.model == 'wrn28':
            semantic_model = RN18_10(semantic=True).to(device)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
            semantic_model.load_state_dict(semantic_checkpoint)
            semantic_model.eval()
            print('load semantic model successfully!')
            clf = WRN28_10(semantic=False).to(device)
            clf = torch.nn.DataParallel(clf)
            checkpoint = torch.load('checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth') # download targeted model 
            clf.load_state_dict(checkpoint)
            clf.eval()
            print('load cls successfully!')
        elif args.model == 'wrn70':
            semantic_model = RN18_10(semantic=True).to(device)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
            semantic_model.load_state_dict(semantic_checkpoint)
            semantic_model.eval()
            print('load semantic model successfully!')
            clf = WRN70_16(semantic=False).to(device)
            clf = torch.nn.DataParallel(clf)
            checkpoint = torch.load('checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth') # download targeted model 
            clf.load_state_dict(checkpoint)
            clf.eval()
            print('load cls successfully!')
        elif args.model == 'rn18':
            checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
            semantic_model = RN18_10(semantic=True).to(device)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.load_state_dict(checkpoint)
            semantic_model.eval()
            print('load semantic model successfully!')
            clf = RN18_10(semantic=False).to(device)
            clf = torch.nn.DataParallel(clf)
            clf.load_state_dict(checkpoint)
            clf.eval()
            print('load cls successfully!')   
        else:
            raise ValueError("Unknown model")
    elif args.data == 'ImageNet':
        args.batch_size = 128
        args.num_workers = 4

        train_dataset = ImageFolder(root='Imagenet/ILSVRC/Data/CLS-LOC/train', 
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        ]))

        with open('data/imagenet_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')

        args.mmd_batch = len(mmd_indices)
        args.mmd_dir = './checkpoint/ImageNet/SAMMD'
        mmd_subset = Subset(train_dataset, mmd_indices)
        mmd_loader = DataLoader(mmd_subset, 
                                batch_size=args.mmd_batch, 
                                shuffle=False, 
                                num_workers=args.num_workers)

        with open('data/imagenet_train_indices.pkl', 'rb') as f:
            train_indices = pickle.load(f)
        print('load train indices successfully!')

        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          num_workers=args.num_workers)
        print('load train dataset successfully!')

        denoiser_dir = './checkpoint/ImageNet/Denoise'
        input_size = [224, 224]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)

        args.num_class = 1000 
        args.epsilon = 4/255
        args.step_size = 1/255
        args.num_steps = 10

        if args.model == 'rn50':
            mma_trainset = torch.load('adv_data/ImageNet/RN50/train_mma_{}_rn50.pth'.format(args.epsilon))
            mmd_mma_trainset = torch.load('adv_data/ImageNet/RN50/train_mma_{}_rn50_mmd.pth'.format(args.epsilon))
            mma_train_loader = DataLoader(mma_trainset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  num_workers=args.num_workers)
            mmd_mma_train_loader = DataLoader(mmd_mma_trainset, 
                                      batch_size=args.mmd_batch, 
                                      shuffle=False, 
                                      num_workers=args.num_workers)

            clf = resnet50(weights="IMAGENET1K_V2").to(device)
            semantic_model = torch.nn.Sequential(*(list(clf.children())[:-1]))

            clf = torch.nn.DataParallel(clf)
            clf.eval()
            print('load cls successfully!')
            
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.eval()
            print(semantic_model)
            print('load semantic model successfully!')
        else:
            raise ValueError("Unknown model")
    elif args.data == 'SVHN':
        args.mmd_dir = './checkpoint/SVHN/SAMMD'
        train_dataset = SVHN('./data/', split='train', download=True, 
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]))
        with open('data/svhn_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')
        args.mmd_batch = len(mmd_indices)
        mmd_subset = Subset(train_dataset, mmd_indices)
        mmd_loader = DataLoader(mmd_subset, 
                          batch_size=args.mmd_batch, 
                          shuffle=False, 
                          num_workers=args.num_workers)
        print('load mmd dataset successfully!')
        with open('data/svhn_train_indices.pkl', 'rb') as f:
            train_indices = pickle.load(f)
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          num_workers=args.num_workers)
        print('load train dataset successfully!')

        denoiser_dir = './checkpoint/SVHN/Denoise'
        input_size = [32, 32]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)

        if args.model == 'wrn28':
            semantic_model = WRN28_10(semantic=True).to(device)
            semantic_checkpoint = torch.load('checkpoint/SVHN/WRN28/wide-resnet-28x10.pth')
            semantic_model.load_state_dict(semantic_checkpoint)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.eval()
            print('load semantic model successfully!')
            clf = WRN28_10(semantic=False).to(device)
            checkpoint = torch.load('checkpoint/SVHN/WRN28/wide-resnet-28x10.pth') # download targeted model 
            clf.load_state_dict(checkpoint)
            clf = torch.nn.DataParallel(clf)
            clf.eval()
            print('load cls successfully!')
    else:
        raise ValueError("Unknown data")

    if not os.path.exists(denoiser_dir):
        os.makedirs(denoiser_dir)

    cudnn.benchmark = True

    optimizer = optim.Adam(denoiser.parameters(), lr=args.lr)

    # extract natural images for mmd testing
    data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
    nat_data = torch.stack(data_only)
    nat_data = nat_data[0:args.batch_size]
    print('the length of nat data is: ', len(nat_data))
    
    if args.data == 'CIFAR10' or args.data == 'SVHN':
        # generating adversarial examples for training
        start_time = time.time()
        print('generating adversarial examples for training')
        args.attack = 'mma'
        mma_trainset = adv_generate(clf, train_loader, device, args)
        mma_train_loader = DataLoader(mma_trainset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=args.num_workers)
        end_time = time.time()
        time_taken = end_time - start_time
        print('finish generation! Time taken: {:.2f} seconds'.format(time_taken))

        print('generating adversarial examples for mmd training')
        start_time = time.time()
        mmd_mma_trainset = adv_generate(clf, mmd_loader, device, args)
        mmd_mma_train_loader = DataLoader(mmd_mma_trainset, 
                                        batch_size=args.mmd_batch, 
                                        shuffle=False, 
                                        num_workers=args.num_workers)
        end_time = time.time()
        time_taken = end_time - start_time
        print('finish generation! Time taken: {:.2f} seconds'.format(time_taken))

    # train mmd
    print('start training the kernel of mmd')
    epsilonOPT, sigma0, sigmaOPT = train_mmd(args, mmd_loader, mmd_mma_train_loader, semantic_model, device)
    ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    print('finish training!')

    mmd_parameters = {
    'ep': ep,
    'sigma0': sigma0,
    'sigma': sigma
    }

    print('mmd parameters are: ', mmd_parameters)

    if not os.path.exists(args.mmd_dir):
        os.makedirs(args.mmd_dir)

    torch.save(mmd_parameters, os.path.join(args.mmd_dir, '{}_mmd_parameters.pth'.format(args.model)))
    print('Parameters of mmd saved successfully.')

    print('start training the denoiser')
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(args, optimizer, epoch)

        # standardly train denoiser
        train_denoiser(args, train_loader, mma_train_loader, 
                        optimizer, epoch, denoiser, semantic_model, clf, 
                        ep, sigma0, sigma, device)

        args.save_freq = 60
        # save checkpoint
        if epoch % args.save_freq == 0:
            if args.data == 'CIFAR10':
                torch.save(denoiser.state_dict(),
                    os.path.join(denoiser_dir, '{}_{}_denoiser_epoch{}_alpha{}_{}.pth'.format(args.data, args.model, epoch, args.alpha, args.index)))
                print('save the denoiser')
            if args.data == 'ImageNet':
                torch.save(denoiser.state_dict(),
                os.path.join(denoiser_dir, '{}_{}_denoiser_epoch{}_{}.pth'.format(args.data, args.model, epoch, args.index)))
                print('save the denoiser')
            if args.data == 'SVHN':
                torch.save(denoiser.state_dict(),
                    os.path.join(denoiser_dir, '{}_{}_denoiser_epoch{}_{}.pth'.format(args.data, args.model, epoch, args.index)))
                print('save the denoiser')
        print('================================================================')

if __name__ == '__main__':
    main()