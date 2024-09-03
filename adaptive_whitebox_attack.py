from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from dataset.cifar10 import CIFAR10
from models.resnet import RN18_10
from models.denoise import Denoise,Conv
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
parser.add_argument('--model', type=str, default='wrn28', choices=['wrn28', 'wrn70'])
parser.add_argument("--mmd-batch", type=int, default=100, help="batch size for mmd training")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--mode', type=str, default='test', help='decide to generate test data or train data')
args = parser.parse_args()

def adaptive_pgd_eot(device,
                 denoiser,
                 semantic_model,
                 clf,
                 X,
                 y,
                 sigma,
                 sigma0,
                 ep,
                 epsilon=8/255,
                 num_steps = 200,
                 eot_steps = 20,
                 step_size=2/255):
    
    images = X.clone().detach().to(device)
    labels = y.clone().detach().to(device)

    nat_feature = semantic_model(images)

    std = 0.25
    mean = 0

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)  # nopep8
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(num_steps):
        grad = torch.zeros_like(adv_images)
        adv_images.requires_grad = True
        for _ in range(eot_steps):
            adv_feature = semantic_model(adv_images)

            S = torch.cat([nat_feature.cpu(), adv_feature.cpu()], 0).cuda()
            Sv = S.view(nat_feature.shape[0] + adv_feature.shape[0], -1)
    
            S = torch.cat([X.cpu(), adv_images.cpu()], 0).cuda()
            S_FEA = S.view(X.shape[0] + adv_images.shape[0], -1)

            _, _, mmd_value = SAMMD_WB(Sv, 100, X.shape[0], S_FEA, 
                                sigma, sigma0, ep, 0.05, device, torch.float)
            
            with torch.enable_grad():
                noise = torch.randn(adv_images.size()) * std + mean
                noise = noise.to(device)
                noise = torch.clamp(noise, 0, 1)
                noisy_data = torch.clamp(adv_images + noise, 0, 1)

            if mmd_value <= 0.05:
                outputs = clf(adv_images)
            else:
                outputs = clf(denoiser(noisy_data))

            # Calculate loss
            cost = loss(outputs, labels)
            
            total_cost = cost + 100*mmd_value

            # Update adversarial images
            grad += torch.autograd.grad(
                total_cost, adv_images, retain_graph=False, create_graph=False
            )[0]

        # (grad/self.eot_iter).sign() == grad.sign()
        adv_images = adv_images.detach() + step_size * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

def adaptive_pgd_eot_generate(denoiser, semantic_model, model, sigma, sigma0, ep, test_loader, device):
    model.eval()
    adv_dataset = AdvDataset()

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_adv = adaptive_pgd_eot(device, denoiser, semantic_model, model, data, target, sigma, sigma0, ep)
            adv_dataset.add(x_adv.cpu(), target.cpu())
            del x_adv, target, data
            torch.cuda.empty_cache()
    return adv_dataset

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

        _, _, mmd_value = SAMMD_WB(Sv, 100, nat_feature.shape[0], S_FEA, 
                                   sigma, sigma0, ep, 0.05, device, torch.float)
        print('MMD value: {:.4f}'.format(mmd_value))

        #add Gaussian noise
        noise = torch.randn(data.size()) * std + mean
        noise = noise.to(device)
        noise = torch.clamp(noise, 0, 1)
        noisy_data = torch.clamp(data + noise, 0, 1)

        if mmd_value <= 0.05:
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
    #test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()

    denoiser_dir = './checkpoint/CIFAR10/Denoise'
    input_size = [32, 32]
    block = Conv
    fwd_out = [64, 128, 256, 256, 256]
    num_fwd = [2, 3, 3, 3, 3]
    back_out = [64, 128, 256, 256]
    num_back = [2, 3, 3, 3]
    denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
    denoiser = torch.nn.DataParallel(denoiser)

    checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
    semantic_model = RN18_10(semantic=True).to(device)
    semantic_model = torch.nn.DataParallel(semantic_model)
    semantic_model.load_state_dict(checkpoint)
    semantic_model.eval()
    print('load semantic model successfully!')

    if args.model == 'wrn28':
        clf_checkpoint = torch.load('checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth')
        clf = WRN28_10(semantic=False).to(device)
        loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir))
        denoiser_ckpt = torch.load('{}/CIFAR10_wrn28_denoiser_epoch60.pth'.format(denoiser_dir))
        PATH_DATA='./adv_data/CIFAR10/WRN28'
    elif args.model == 'wrn70':
        clf_checkpoint = torch.load('checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth')
        clf = WRN70_16(semantic=False).to(device)
        loaded_parameters = torch.load('{}/wrn70_mmd_parameters.pth'.format(args.mmd_dir))
        denoiser_ckpt = torch.load('{}/CIFAR10_wrn70_denoiser_epoch60.pth'.format(denoiser_dir))
        PATH_DATA='./adv_data/CIFAR10/WRN70'

    clf = torch.nn.DataParallel(clf)
    clf.load_state_dict(clf_checkpoint)
    clf.eval()
    print('load cls successfully!')

    train_dataset = train_loader.dataset
        
    ep = loaded_parameters['ep']
    sigma0 = loaded_parameters['sigma0']
    sigma = loaded_parameters['sigma']

    # load checkpoints of denoiser
    denoiser.load_state_dict(denoiser_ckpt)

    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)

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
    
    # print('==> Generate adversarial sample')
    # adaptive_adv_dataset = adaptive_pgd_eot_generate(denoiser, semantic_model, clf, sigma, sigma0, ep, test_loader, device)
    # print('==> Save adversarial sample')
    # torch.save(adaptive_adv_dataset, os.path.join(PATH_DATA, f'{args.mode}_{args.model}_PGDEOT_adaptive.pth'))
    adaptive_adv_dataset = torch.load(os.path.join(PATH_DATA, f'{args.mode}_{args.model}_PGDEOT_adaptive.pth'))
    adaptive_test_loader = DataLoader(adaptive_adv_dataset, batch_size=args.mmd_batch, shuffle=False)

    print('=====================Adaptive PGDEOT Accuracy====================')
    eval_test(denoiser, clf, device, adaptive_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

if __name__ == '__main__':
    main()