import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable
from scipy import signal
from torchattacks import PGD, CW, EOTPGD, AutoAttack
# from autoattack.autoattack import AutoAttack
from torch.utils.data import Dataset

def parse_fraction(fraction_string):
    if '/' in fraction_string:
        numerator, denominator = fraction_string.split('/')
        return float(numerator) / float(denominator)
    return float(fraction_string)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)

def adjust_learning_rate(args, optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= args.second_decay:
        lr = args.lr * 0.01
    elif epoch >= args.first_decay:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    #print('x_norm has a shape ',x_norm.shape)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
        #print('y_norm has a shape ',y_norm.shape)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def Wild_bootsrap_process(data_length, Num_trails):
    #ln = 20.0
    #ln = 0.2
    ln = 0.2
    ar = np.exp(-1/ln)
    variance = 1-np.exp(-2/ln)

    w = np.sqrt(variance) * np.random.randn(data_length, Num_trails)
    a = np.array([1,-1 * ar])
    process = signal.lfilter(np.array([1]), a, w)
    return process

def bootsrap_null(data_length, Num_trails, Kx, Ky, Kxy, alpha, device, dtype):

    process = MatConvert(Wild_bootsrap_process(data_length, Num_trails), device, dtype)
    testStatMat = Kx + Ky - 2*Kxy
    testStat = testStatMat.mean()

    testStat_tem = torch.zeros(Num_trails).to(device, dtype)
    count = 0
    for kk in range(Num_trails):
        mn = process[:,kk].mean()
        matWB = (process[:,kk] - mn).unsqueeze(1).matmul((process[:,kk] - mn).unsqueeze(0))
        # represent the estimated statistics sampled from wild bootstrap
        testStat_tem[kk] = (testStatMat * matWB).mean()
        if testStat_tem[kk] >= testStat:
            count = count + 1
        if count > np.ceil(Num_trails * alpha):
            # do not reject the null hypothesisS
            h = 0
            threshold = "NaN"
            break
        else:
            # reject the null hypothesis
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(testStat_tem.cpu().detach().numpy())
        threshold = S_mmd_vector[int(np.ceil(Num_trails * (1 - alpha)))]
    return h, threshold, testStat

def SAMMD_WB(Fea, N_per, N_te, Fea_org, sigma, sigma0, epsilon, alpha, device, dtype):

    X = Fea[0:N_te, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[N_te:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:N_te, :] # fetch the original sample 1
    Y_org = Fea_org[N_te:, :] # fetch the original sample 2

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)

    Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0) -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
    Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0) -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
    Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0) -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)

    return bootsrap_null(N_te, N_per, Kx, Ky, Kxy, alpha, device, dtype)

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error_var!!'+str(V1))
        print(Kx.shape)
    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def craft_adversarial_example(model, 
                              x_natural, 
                              y,
                              step_size=2/255, 
                              epsilon=8/255, 
                              perturb_steps=10,
                              num_classes=10,
                              mode='pgd'):
    if mode == 'pgd':
        attack = PGD(model, 
                     eps=epsilon, 
                     alpha=step_size, 
                     steps=perturb_steps, 
                     random_start=True)
        x_adv = attack(x_natural, y)
    elif mode == 'aa_l2':
        attack = AutoAttack(model, 
                            norm='L2', 
                            eps=0.5, 
                            version='rand',
                            n_classes = num_classes)
        x_adv = attack(x_natural, y)
    elif mode == 'aa':
        attack = AutoAttack(model, 
                            norm='Linf', 
                            eps=8/255, 
                            version='rand',
                            n_classes=num_classes)
        x_adv = attack(x_natural, y)
    elif mode == 'eotpgd':
        attack = EOTPGD(model,
                        eps=epsilon,
                        alpha=step_size,
                        steps=perturb_steps,
                        eot_iter=2)
        x_adv = attack(x_natural, y)
    elif mode == 'mma':
        x_adv = mma(model,
                    data=x_natural,
                    target=y,
                    epsilon=epsilon,
                    step_size=step_size,
                    num_steps=perturb_steps,
                    category='Madry',
                    rand_init=True,
                    k=3,
                    num_classes=num_classes)
    elif mode == 'cw':
        attack = CW(model, c=epsilon, kappa=0, steps=perturb_steps, lr=0.01)
        x_adv = attack(x_natural, y)
    return x_adv

def mma(model, 
        data, 
        target, 
        epsilon, 
        step_size, 
        num_steps, 
        category, 
        rand_init, 
        k, 
        num_classes):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    logits = model(data)
    target_onehot = torch.zeros(target.size() + (len(logits[0]),))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    index = torch.argsort(logits - 10000 * target_var)[:, num_classes - k:]

    x_adv_set = []
    loss_set = []
    for i in range(k):
        x_adv_0 = x_adv.clone().detach()
        for j in range(num_steps):
            x_adv_0.requires_grad_()
            output1 = model(x_adv_0)
            model.zero_grad()
            with torch.enable_grad():
                loss_adv0 = mm_loss(output1, target, index[:, i], num_classes=num_classes)
            loss_adv0.backward()
            eta = step_size * x_adv_0.grad.sign()
            x_adv_0 = x_adv_0.detach() + eta
            x_adv_0 = torch.min(torch.max(x_adv_0, data - epsilon), data + epsilon)
            x_adv_0 = torch.clamp(x_adv_0, 0.0, 1.0)

        pipy = mm_loss_train(model(x_adv_0), target, index[:, i], num_classes=num_classes)
        loss_set.append(pipy.view(len(pipy), -1))
        x_adv_set.append(x_adv_0)

    loss_pipy = loss_set[0]
    for i in range(k - 1):
        loss_pipy = torch.cat((loss_pipy, loss_set[i + 1]), 1)

    index_choose = torch.argsort(loss_pipy)[:, -1]

    adv_final = torch.zeros(x_adv.size()).cuda()
    for i in range(len(index_choose)):
        adv_final[i, :, :, :] = x_adv_set[index_choose[i]][i]

    return adv_final

# loss for MM AT
def mm_loss_train(output, target, target_choose, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    return other-real

# loss for MM Attack
def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

class AdvDataset(Dataset):
    def __init__(self):
        self.data = []
        self.targets = []

    def add(self, x_advs, targets):
        for x_adv, target in zip(x_advs, targets):
            self.data.append(x_adv)
            self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def adv_generate(model, test_loader, device, args):
    model.eval()
    adv_dataset = AdvDataset()

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if args.attack == 'pgd':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='pgd')
            if args.attack == 'mma':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='mma')
            if args.attack == 'aa':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='aa')
            if args.attack == 'aa_l2':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='aa_l2')
            if args.attack == 'eotpgd':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='eotpgd')
            if args.attack == 'cw':
                x_adv = craft_adversarial_example(model, 
                                                  data, 
                                                  target,
                                                  step_size=args.step_size, 
                                                  epsilon=args.epsilon, 
                                                  perturb_steps=args.num_steps,
                                                  num_classes=args.num_class,
                                                  mode='cw')
            adv_dataset.add(x_adv.cpu(), target.cpu())
            del x_adv, target, data
            torch.cuda.empty_cache()
    return adv_dataset