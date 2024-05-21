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
parser.add_argument('--model', type=str, default='wrn28', choices=['wrn28', 'rn50'])
parser.add_argument("--mmd-batch", type=int, default=100, help="batch size for mmd training")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epsilon', default=8/255, type=parse_fraction, help='perturbation')
parser.add_argument('--data', type=str, default='CIFAR10', help='data source', choices=['CIFAR10', 'ImageNet'])
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

        h1, _, _ = SAMMD_WB(Sv, 100, nat_feature.shape[0], S_FEA, 
                                   sigma, sigma0, ep, 0.05, device, torch.float)

        #add Gaussian noise
        noise = torch.randn(data.size()) * std + mean
        noise = noise.to(device)
        noise = torch.clamp(noise, 0, 1)
        noisy_data = torch.clamp(data + noise, 0, 1)

        if h1 == 0:
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

    loaded_parameters = torch.load('{}/wrn28_mmd_parameters.pth'.format(args.mmd_dir, args.model))
    ep = loaded_parameters['ep']
    sigma0 = loaded_parameters['sigma0']
    sigma = loaded_parameters['sigma']

    denoiser_ckpt = torch.load('{}/CIFAR10_wrn28_denoiser_epoch60.pth'.format(denoiser_dir))
    denoiser.load_state_dict(denoiser_ckpt)

    aa_testset = torch.load('adv_data/CIFAR10/WRN28/test_aa_{}_wrn28.pth'.format(args.epsilon))
    aa_test_loader = DataLoader(aa_testset, batch_size=args.batch_size, shuffle=False)

    aa_l2_testset = torch.load('adv_data/CIFAR10/WRN28/test_aa_l2_0.5_wrn28.pth')
    aa_l2_test_loader = DataLoader(aa_l2_testset, batch_size=args.batch_size, shuffle=False)

    print('=====================Natural Accuracy===================')
    eval_test(clf, semantic_model, denoiser, device, test_loader, nat_data, ep, sigma0, sigma)
    print('======================AA L-infinity Accuracy============')
    eval_test(clf, semantic_model, denoiser, device, aa_test_loader, nat_data, ep, sigma0, sigma)
    print('======================AA L-2 Accuracy===================')
    eval_test(clf, semantic_model, denoiser, device, aa_l2_test_loader, nat_data, ep, sigma0, sigma)

if __name__ == '__main__':
    main()