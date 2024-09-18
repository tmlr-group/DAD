from __future__ import print_function
import os
from models.denoise import Denoise,Conv
import torch.backends.cudnn as cudnn
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
parser.add_argument('--model', type=str, default='wrn28', choices=['wrn28', 'wrn70', 'rn50'])
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epsilon', default=8/255, type=parse_fraction, help='perturbation')
parser.add_argument('--data', type=str, default='CIFAR10', help='data source', choices=['CIFAR10', 'ImageNet'])
parser.add_argument('--index', type=int, default=1, help='index of the model')
parser.add_argument('--white-box', action='store_true', default=False, help='white-box attack or non-white-box attack')
args = parser.parse_args()

def eval_test(clf, semantic_model, denoiser, device, test_loader, nat_data, ep, sigma0, sigma):
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

        if mmd_value < 0.05:
            logits_out = clf(data)
        else:
            X_puri = denoiser(noisy_data)
            logits_out = clf(X_puri)
        
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

    print('Test Accuracy: {}/{} ({:.2f}%)'.format(correct,
        len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # settings for CIFAR10
    setup_seed(1)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.data == "CIFAR10":
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
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

        if args.model == 'wrn28':
            semantic_checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
            semantic_model = RN18_10(semantic=True).to(device)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.load_state_dict(semantic_checkpoint)
            semantic_model.eval()
            print('load semantic model successfully!')

            checkpoint = torch.load('checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth')
            clf = WRN28_10(semantic=False).to(device)
            clf = torch.nn.DataParallel(clf)
            clf.load_state_dict(checkpoint)
            clf.eval()
            print('load cls successfully!')

            loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir))
            ep = loaded_parameters['ep']
            sigma0 = loaded_parameters['sigma0']
            sigma = loaded_parameters['sigma']

            adv_dir = './adv_data/CIFAR10/WRN28'

        if args.model == 'wrn70':
            semantic_checkpoint = torch.load('checkpoint/CIFAR10/RN18/resnet-18.pth')
            semantic_model = RN18_10(semantic=True).to(device)
            semantic_model = torch.nn.DataParallel(semantic_model)
            semantic_model.load_state_dict(semantic_checkpoint)
            semantic_model.eval()
            print('load semantic model successfully!')

            checkpoint = torch.load('checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth')
            clf = WRN70_16(semantic=False).to(device)
            clf = torch.nn.DataParallel(clf)
            clf.load_state_dict(checkpoint)
            clf.eval()
            print('load cls successfully!')

            loaded_parameters = torch.load('{}/wrn70_mmd_parameters.pth'.format(args.mmd_dir))
            ep = loaded_parameters['ep']
            sigma0 = loaded_parameters['sigma0']
            sigma = loaded_parameters['sigma']

            adv_dir = './adv_data/CIFAR10/WRN70'

        train_dataset = train_loader.dataset

        # create subsets
        with open('data/cifar10_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')
        mmd_subset = Subset(train_dataset, mmd_indices)
        print('load mmd dataset successfully!')
        data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
        nat_data = torch.stack(data_only)

        cudnn.benchmark = True

        denoiser_ckpt = torch.load('{}/CIFAR10_{}_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.model, args.index))
        denoiser.load_state_dict(denoiser_ckpt)

    if args.data == 'ImageNet':
        with open('data/imagenet_mmd_indices.pkl', 'rb') as f:
            mmd_indices = pickle.load(f)
        print('load mmd indices successfully!')

        train_dataset = ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train', 
                                        transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            ]))
        mmd_subset = Subset(train_dataset, mmd_indices)
        data_only = [mmd_subset[i][0] for i in range(len(mmd_subset))]
        nat_data = torch.stack(data_only)

        test_loader = ImageNet(test_batch_size=args.batch_size, num_workers=4).test_data()

        denoiser_dir = './checkpoint/ImageNet/Denoise'
        input_size = [224, 224]
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        denoiser = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back).to(device)
        denoiser = torch.nn.DataParallel(denoiser)
        denoiser_ckpt = torch.load('{}/ImageNet_{}_denoiser_epoch60_{}.pth'.format(denoiser_dir, args.model, args.index))
        denoiser.load_state_dict(denoiser_ckpt)

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
            ep = loaded_parameters['ep']
            sigma0 = loaded_parameters['sigma0']
            sigma = loaded_parameters['sigma']

    print('=====================Natural Accuracy===================')
    eval_test(clf, semantic_model, denoiser, device, test_loader, nat_data, ep, sigma0, sigma)
    print('=====================Whitebox Robust Accuracy on AutoAttack L_inf===================')
    aa_testset = torch.load('{}/test_{}_aa_{}_{}_{}.pth'.format(adv_dir, args.white_box, args.epsilon, args.model, args.index))
    aa_test_loader = DataLoader(aa_testset, batch_size=args.batch_size, shuffle=False)
    eval_test(clf, semantic_model, denoiser, device, aa_test_loader, nat_data, ep, sigma0, sigma)
    print('=====================Whitebox Robust Accuracy on AutoAttack L_2===================')
    aa_testset = torch.load('{}/test_{}_aa_l2_{}_{}_{}.pth'.format(adv_dir, args.white_box, args.epsilon, args.model, args.index))
    aa_test_loader = DataLoader(aa_testset, batch_size=args.batch_size, shuffle=False)
    eval_test(clf, semantic_model, denoiser, device, aa_test_loader, nat_data, ep, sigma0, sigma)

if __name__ == '__main__':
    main()