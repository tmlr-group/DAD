import argparse
import os
import torch
from dataset.cifar10 import CIFAR10
from dataset.imagenet import ImageNet
from dataset.svhn import SVHN
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from models.resnet import RN18_10, RN50_10
from models.wide_resnet import WRN28_10, WRN70_16
from models.swin import swin_t
from torchvision.models import resnet50
from models.denoise import Denoise,Conv
from utils import *
import pickle

parser = argparse.ArgumentParser(description='PyTorch DAD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoint/CIFAR10/Denoise',
                    help='directory of model for saving checkpoint')
parser.add_argument('--mmd-dir', default='./checkpoint/CIFAR10/SAMMD',
                    help='directory of mmd for saving checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument("--alpha", type=float, default=1e-2, help="regularization term")
parser.add_argument('--model', type=str, default='wrn28', choices=['rn18', 'wrn28', 'wrn70', 'rn50', 'swin'])
parser.add_argument("--mmd-batch", type=int, default=100, help="batch size for mmd training")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--data', type=str, default='CIFAR10', help='data source', choices=['CIFAR10', 'ImageNet', 'SVHN'])
parser.add_argument('--attack',type=str,default='mma',help='select attack setting')
parser.add_argument('--mode', type=str, default='test', help='decide to generate test data or train data')
parser.add_argument('--epsilon', default=8/255, type=parse_fraction, help='perturbation')
parser.add_argument('--num-steps', default=100, type=int, help='perturb number of steps')
parser.add_argument('--num-class', default=10, help='number of classes')
parser.add_argument('--step-size', default=2/255, type=parse_fraction, help='perturb step size')
parser.add_argument('--index', default=1, type=int, help='index of model')
parser.add_argument('--white-box', action='store_true', default=False, help='white-box attack or non-white-box attack')
parser.add_argument('--generate', action='store_true', default=False, help='generate adv or not')
parser.add_argument('--norm', type=str, default='l_inf', help='l_norm', choices=['l_inf', 'l_2'])
parser.add_argument('--threshold', type=float, default=0.05, help='threshold for detecting AEs')
parser.add_argument('--fpr', type=float, default=0.05, help='fpr for detecting AEs')
parser.add_argument('--type', type=str, default='vanilla', choices=['vanilla', 'trades', 'mart', 'awp'])
args = parser.parse_args()

class Adaptive_Whitebox_Model(nn.Module):
    def __init__(self, index, device, args):
        super().__init__()
        self.args = args
        self.device = device
        self.index = index
        self.noise_mean = 0.0
        self.noise_std = 0.25

        if args.model == "rn18":
            model = RN18_10(semantic=False).to(device)
            model_path = './checkpoint/CIFAR10/RN18/resnet-18.pth'
            ckpt = torch.load(model_path)
            self.model = torch.nn.DataParallel(model)
            model.load_state_dict(ckpt)
        if args.model == 'wrn28':
            model = WRN28_10(semantic=False).to(device)
            model_path = './checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth'
            ckpt = torch.load(model_path)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(ckpt)
        if args.model == 'wrn70':
            self.model = WRN70_16(semantic=False).to(device)
            model_path = './checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth'
            ckpt = torch.load(model_path)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(ckpt)
        if args.model == 'rn50':
            if args.data == 'ImageNet':
                model = resnet50(weights="IMAGENET1K_V2").to(device)
                model = torch.nn.DataParallel(model)
            elif args.data == 'CIFAR10':
                model = RN50_10(semantic=False).to(device)
                model_path = './checkpoint/CIFAR10/RN50/resnet-50.pth'
                ckpt = torch.load(model_path)
                model = torch.nn.DataParallel(model)
        if args.model == 'swin':
                model = swin_t(window_size=4,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1)).to(device)
                model_path = './checkpoint/CIFAR10/SWIN/swin-4-ckpt.pth'
                ckpt = torch.load(model_path)
                model.load_state_dict(ckpt['model'])
        # image classifier
        self.classifier = model

        # denoiser
        if args.data == 'CIFAR10':
            denoiser_dir = './checkpoint/CIFAR10/Denoise'
            input_size = [32, 32]
            block = Conv
            fwd_out = [64, 128, 256, 256, 256]
            num_fwd = [2, 3, 3, 3, 3]
            back_out = [64, 128, 256, 256]
            num_back = [2, 3, 3, 3]
            denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
            denoiser = torch.nn.DataParallel(denoiser)
            denoiser_ckpt = torch.load('{}/{}_{}_denoiser_epoch60_alpha{}_{}.pth'.format(denoiser_dir, args.data, args.model, args.alpha, index))
            denoiser.load_state_dict(denoiser_ckpt)
        if args.data == 'ImageNet':
            denoiser_dir = './checkpoint/ImageNet/Denoise'
            input_size = [224, 224]
            block = Conv
            fwd_out = [64, 128, 256, 256, 256]
            num_fwd = [2, 3, 3, 3, 3]
            back_out = [64, 128, 256, 256]
            num_back = [2, 3, 3, 3]
            denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
            denoiser = torch.nn.DataParallel(denoiser)
            denoiser_ckpt = torch.load('{}/{}_{}_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.data, args.model, index))
            denoiser.load_state_dict(denoiser_ckpt)
        self.denoiser = denoiser

    def forward(self, x):
        #add Gaussian noise
        noise = torch.randn(x.size()) * self.noise_std + self.noise_mean
        noise = noise.to(self.device)
        noise = torch.clamp(noise, 0, 1)
        noisy_data = torch.clamp(x + noise, 0, 1)

        x_denoised = self.denoiser(noisy_data)
        out = self.classifier(x_denoised)
        return out

def main():
    setup_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('==> Load Test Data')
    if args.data == "CIFAR10":
        train_loader = CIFAR10(train_batch_size=500).train_data()
        test_loader = CIFAR10(test_batch_size=128).test_data()
    if args.data == "ImageNet":
        args.num_class = 1000 
        args.epsilon = 4/255
        args.step_size = 1/255
        args.num_steps = 10
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
                                batch_size=128, 
                                shuffle=False, 
                                num_workers=3)

        with open('data/imagenet_train_indices.pkl', 'rb') as f:
            train_indices = pickle.load(f)
        print('load train indices successfully!')

        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, 
                          batch_size=128, 
                          shuffle=False, 
                          num_workers=3)
        print('load train dataset successfully!')
        test_loader = ImageNet(test_batch_size=100, num_workers=3).test_data()

    if args.model == "rn18":
        model = RN18_10(semantic=False).to(device)
        model_path = './checkpoint/CIFAR10/RN18/resnet-18.pth'
        ckpt = torch.load(model_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt)
    if args.model == 'wrn28':
        model = WRN28_10(semantic=False).to(device)
        model_path = './checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth'
        ckpt = torch.load(model_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt)
    if args.model == 'wrn70':
        model = WRN70_16(semantic=False).to(device)
        model_path = './checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth'
        ckpt = torch.load(model_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt)
    if args.model == 'rn50':
        if args.data == 'ImageNet':
            model = resnet50(weights="IMAGENET1K_V2").to(device)
            model = torch.nn.DataParallel(model)
        elif args.data == 'CIFAR10':
            model = RN50_10(semantic=False).to(device)
            model_path = './checkpoint/CIFAR10/RN50/resnet-50.pth'
            ckpt = torch.load(model_path)
            model = torch.nn.DataParallel(model)
    if args.model == 'swin':
            model = swin_t(window_size=4, num_classes=10, downscaling_factors=(2,2,2,1)).to(device)
            model_path = './checkpoint/CIFAR10/SWIN/swin-4-ckpt.pth'
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['model'])

    if args.model == "rn18":
        PATH_DATA='./adv_data/CIFAR10/RN18'
    if args.model == 'wrn28':
        PATH_DATA='./adv_data/CIFAR10/WRN28'
    if args.model == 'wrn70':
        PATH_DATA='./adv_data/CIFAR10/WRN70'
    if args.model == 'rn50':
        if args.data == 'ImageNet':
            PATH_DATA='./adv_data/ImageNet/RN50'
        elif args.data == 'CIFAR10':
            PATH_DATA='./adv_data/CIFAR10/RN50'
    if args.model == 'swin':
            PATH_DATA='./adv_data/CIFAR10/SWIN'

    print('==> Load Model')
    if args.white_box:
        model = Adaptive_Whitebox_Model(args.index, device, args)
    model.eval()

    print('==> Generate adversarial sample')
    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)

    if args.mode == 'test':
        adv_dataset = adv_generate(model, test_loader, device, args)
        torch.save(adv_dataset, os.path.join(PATH_DATA, f'{args.mode}_{args.white_box}_{args.attack}_{args.epsilon}_{args.model}_{args.index}.pth'))
    if args.mode == 'train':
        adv_dataset = adv_generate(model, train_loader, device, args)
        torch.save(adv_dataset, os.path.join(PATH_DATA, f'{args.mode}_{args.attack}_{args.epsilon}_{args.model}.pth'))
        if args.data == 'ImageNet':
            adv_mmd_dataset = adv_generate(model, mmd_loader, device, args)
            torch.save(adv_mmd_dataset, os.path.join(PATH_DATA, f'{args.mode}_{args.attack}_{args.epsilon}_{args.model}_mmd.pth'))

if __name__ == '__main__':
    main()
