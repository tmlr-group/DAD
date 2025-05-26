from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from dataset.cifar10 import CIFAR10
from models.resnet import RN18_10
from models.denoise import Denoise,Conv
from utils import *
from adv_generator import *

def adaptive_pgd_eot_l_inf(device,
                 denoiser,
                 semantic_model,
                 clf,
                 X,
                 y,
                 sigma,
                 sigma0,
                 ep,
                 args):
    
    epsilon = args.epsilon
    num_steps = args.num_steps
    step_size = args.step_size
    eot_steps = 20
    print('epsilon: ', epsilon)
    print('num_steps: ', num_steps)
    print('eot_steps: ', eot_steps)
    print('step_size: ', step_size)
    
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
            
            total_cost = mmd_value + args.alpha*cost

            # Update adversarial images
            grad += torch.autograd.grad(
                total_cost, adv_images, retain_graph=False, create_graph=False
            )[0]

        # (grad/self.eot_iter).sign() == grad.sign()
        adv_images = adv_images.detach() + step_size * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def adaptive_pgd_eot_l2(device,
                 denoiser,
                 semantic_model,
                 clf,
                 X,
                 y,
                 sigma,
                 sigma0,
                 ep,
                 args):
    
    epsilon = args.epsilon
    num_steps = args.num_steps
    step_size = args.step_size
    eot_steps = 20
    print('epsilon: ', epsilon)
    print('num_steps: ', num_steps)
    print('eot_steps: ', eot_steps)
    print('step_size: ', step_size)
    
    images = X.clone().detach().to(device)
    labels = y.clone().detach().to(device)

    nat_feature = semantic_model(images)

    std = 0.25
    mean = 0

    loss = nn.CrossEntropyLoss()

    batch_size = len(images)
    adv_images = images.clone().detach()

    delta = torch.empty_like(adv_images).normal_()
    d_flat = delta.view(adv_images.size(0), -1)
    n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
    r = torch.zeros_like(n).uniform_(0, 1)
    delta *= r / n * epsilon
    adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

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
        
        grad = grad / eot_steps
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=2, dim=1)+ 1e-10
        )  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        adv_images = adv_images.detach() + step_size * grad

        delta = adv_images - images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def adaptive_pgd_eot_generate(denoiser, semantic_model, model, sigma, sigma0, ep, test_loader, device):
    model.eval()
    adv_dataset = AdvDataset()

    if args.data == 'CIFAR10' or args.data == 'SVHN':
        args.step_size = 2/255
        args.num_steps = 200
        args.epsilon = 8/255
    if args.data == 'ImageNet':
        args.step_size = 1/255
        args.num_steps = 20
        args.epsilon = 4/255
    if args.norm == 'l_2':
        args.epsilon = 0.5

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.norm == 'l_inf':
                x_adv = adaptive_pgd_eot_l_inf(device, denoiser, semantic_model, model, data, target, sigma, sigma0, ep, args)
            if args.norm == 'l_2':
                x_adv = adaptive_pgd_eot_l2(device, denoiser, semantic_model, model, data, target, sigma, sigma0, ep, args)
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

        #add Gaussian noise
        noise = torch.randn(data.size()) * std + mean
        noise = noise.to(device)
        noise = torch.clamp(noise, 0, 1)
        noisy_data = torch.clamp(data + noise, 0, 1)

        if mmd_value <= args.threshold:
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

    if args.data == "CIFAR10":
        args.num_class = 10
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
        train_dataset = train_loader.dataset
        test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()

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

         # create subsets
        with open('data/cifar10_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')
        args.mmd_batch = len(mmd_indices)
        mmd_subset = Subset(train_dataset, mmd_indices)
        print('load mmd dataset successfully!')
        
        data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
        nat_data = torch.stack(data_only)

        if args.model == 'wrn28':
            clf_checkpoint = torch.load('checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth')
            clf = WRN28_10(semantic=False).to(device)
            loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir))
            denoiser_ckpt = torch.load('{}/CIFAR10_wrn28_denoiser_epoch60_alpha{}_{}.pth'.format(denoiser_dir, args.alpha, args.index))
            PATH_DATA='./adv_data/CIFAR10/WRN28'
        elif args.model == 'wrn70':
            clf_checkpoint = torch.load('checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth')
            clf = WRN70_16(semantic=False).to(device)
            loaded_parameters = torch.load('{}/wrn70_mmd_parameters.pth'.format(args.mmd_dir))
            denoiser_ckpt = torch.load('{}/CIFAR10_wrn70_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.index))
            PATH_DATA='./adv_data/CIFAR10/WRN70'
        clf = torch.nn.DataParallel(clf)
        clf.load_state_dict(clf_checkpoint)
        clf.eval()
        print('load cls successfully!')
    if args.data == "ImageNet":
        args.num_class = 1000
        train_dataset = ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train', 
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
        data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
        nat_data = torch.stack(data_only)
        nat_data = nat_data[0:args.batch_size]
        print('the length of nat data is: ', len(nat_data))

        test_loader = ImageNet(test_batch_size=args.batch_size, num_workers=3).test_data()

        denoiser_dir = './checkpoint/ImageNet/Denoise'
        input_size = [224, 224]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)
        denoiser_ckpt = torch.load('{}/ImageNet_rn50_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.index))

        if args.model == 'rn50':
            clf = resnet50(weights="IMAGENET1K_V2").to(device)
            semantic_model = torch.nn.Sequential(*(list(clf.children())[:-1]))
            clf = torch.nn.DataParallel(clf)
            clf.eval()
            print('load cls successfully!')
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.eval()
            print('load semantic model successfully!')
            loaded_parameters = torch.load('checkpoint/ImageNet/SAMMD/rn50_mmd_parameters.pth')
            PATH_DATA='./adv_data/ImageNet/RN50'
    if args.data == "SVHN":
        args.mmd_dir = './checkpoint/SVHN/SAMMD'
        args.num_class = 10
        train_loader = SVHN(train_batch_size=args.batch_size).train_data()
        train_dataset = train_loader.dataset
        test_loader = SVHN(test_batch_size=args.batch_size).test_data()

        denoiser_dir = './checkpoint/SVHN/Denoise'
        input_size = [32, 32]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)

        checkpoint = torch.load('checkpoint/SVHN/WRN28/wide-resnet-28x10.pth')
        semantic_model = WRN28_10(semantic=True).to(device)
        semantic_model.load_state_dict(checkpoint)
        semantic_model = torch.nn.DataParallel(semantic_model)
        semantic_model.eval()
        print('load semantic model successfully!')

         # create subsets
        with open('data/svhn_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')
        args.mmd_batch = len(mmd_indices)
        mmd_subset = Subset(train_dataset, mmd_indices)
        print('load mmd dataset successfully!')
        
        data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
        nat_data = torch.stack(data_only)

        if args.model == 'wrn28':
            clf_checkpoint = torch.load('checkpoint/SVHN/WRN28/wide-resnet-28x10.pth')
            clf = WRN28_10(semantic=False).to(device)
            loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir))
            denoiser_ckpt = torch.load('{}/SVHN_wrn28_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.index))
            PATH_DATA='./adv_data/SVHN/WRN28'
        
        clf.load_state_dict(clf_checkpoint)
        clf = torch.nn.DataParallel(clf)
        clf.eval()
        
    ep = loaded_parameters['ep']
    sigma0 = loaded_parameters['sigma0']
    sigma = loaded_parameters['sigma']

    # load checkpoints of denoiser
    denoiser.load_state_dict(denoiser_ckpt)

    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)

    cudnn.benchmark = True
    
    if args.generate:
        print('==> Generate adversarial sample')
        adaptive_adv_dataset = adaptive_pgd_eot_generate(denoiser, semantic_model, clf, sigma, sigma0, ep, test_loader, device)
        print('==> Save adversarial sample')
        torch.save(adaptive_adv_dataset, os.path.join(PATH_DATA, f'{args.mode}_{args.norm}_{args.model}_PGDEOT_adaptive_alpha{args.alpha}.pth'))

    print('=====================Natural Accuracy===================')
    eval_test(denoiser, clf, device, test_loader, nat_data, semantic_model, sigma, sigma0, ep)

    print('=====================Adaptive PGDEOT L_inf Accuracy====================')
    l_inf_adaptive_adv_dataset = torch.load(os.path.join(PATH_DATA, f'{args.mode}_l_inf_{args.model}_PGDEOT_adaptive_alpha{args.alpha}.pth'))
    l_inf_adaptive_test_loader = DataLoader(l_inf_adaptive_adv_dataset, batch_size=args.batch_size, shuffle=False)
    eval_test(denoiser, clf, device, l_inf_adaptive_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

    # print('=====================Adaptive PGDEOT L_2 Accuracy====================')
    # l_2_adaptive_adv_dataset = torch.load(os.path.join(PATH_DATA, f'{args.mode}_l_2_{args.model}_PGDEOT_adaptive.pth'))
    # l_2_adaptive_test_loader = DataLoader(l_2_adaptive_adv_dataset, batch_size=args.mmd_batch, shuffle=False)
    # eval_test(denoiser, clf, device, l_2_adaptive_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

    # print('=====================Adaptive AutoAttack Accuracy====================')
    # l_inf_adaptive_adv_dataset = torch.load(os.path.join(PATH_DATA, f'{args.mode}_True_aa_{args.epsilon}_{args.model}.pth'))
    # l_inf_adaptive_test_loader = DataLoader(l_inf_adaptive_adv_dataset, batch_size=args.batch_size, shuffle=False)
    # eval_test(denoiser, clf, device, l_inf_adaptive_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

    # print('=====================Adaptive AutoAttack L_2 Accuracy====================')
    # l_2_adaptive_adv_dataset = torch.load(os.path.join(PATH_DATA, f'{args.mode}_True_aa_l2_{args.epsilon}_{args.model}.pth'))
    # l_2_adaptive_test_loader = DataLoader(l_2_adaptive_adv_dataset, batch_size=args.mmd_batch, shuffle=False)
    # eval_test(denoiser, clf, device, l_2_adaptive_test_loader, nat_data, semantic_model, sigma, sigma0, ep)

    print("=================================================================")

if __name__ == '__main__':
    main()