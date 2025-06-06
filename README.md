# One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy (ICML 2025)

 #### Abstract
*Statistical adversarial data detection* (SADD) detects whether an upcoming batch contains *adversarial examples* (AEs) by measuring the distributional discrepancies between *clean examples* (CEs) and AEs. In this paper, we explore the strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named ***D***istributional-discrepancy-based ***A***dversarial ***D***efense (DAD). In the training phase, DAD first optimizes the test power of the *maximum mean discrepancy* (MMD) to derive MMD-OPT, which is *a stone that kills two birds*. MMD-OPT first serves as a *guiding signal* to minimize the distributional discrepancy between CEs and AEs to train a denoiser. Then, it serves as a *discriminator* to differentiate CEs and AEs during inference. Overall, in the inference stage, DAD consists of a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DAD outperforms current *state-of-the-art* (SOTA) defense methods by *simultaneously* improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.

#### Figure 1: The illustration of our method.
![pipeline](images/pipeline.jpg)

### Installation
The code is tested with Python 3.9. To install the requried packages, run:
```
pip install -r requirements.txt
```

### Pre-trained Classifiers
The checkpoint of pre-trained classifiers on CIFAR-10 should be put in 
```checkpoint/CIFAR10/[your model name]```. For example, the checkpoint of pre-trained ```WideResNet-28-10``` on CIFAR-10 should be put in ```checkpoint/CIFAR10/WRN28```.
- The training recipe of ```ResNet``` and ```WideResNet``` on CIFAR-10 follows the [GitHub](https://github.com/meliketoy/wide-resnet.pytorch). 

- To train ```ResNet``` and ```WideResNet```:
```
git clone https://github.com/meliketoy/wide-resnet.pytorch.git

# train a WRN-28-10
python3 main.py --lr 0.1 --net_type 'wide-resnet' --depth 28 --widen_factor 10 --dataset 'cifar10'

# train a WRN-70-16
python3 main.py --lr 0.1 --net_type 'wide-resnet' --depth 70 --widen_factor 16 --dataset 'cifar10'

# train a RN-18
python3 main.py --lr 0.1 --net_type 'resnet' --depth 18 --dataset 'cifar10'

# train a RN-50
python3 main.py --lr 0.1 --net_type 'resnet' --depth 50 --dataset 'cifar10'
```
- The training recipe of ```Swin-Transformer``` on CIFAR-10 follows the [GitHub](https://github.com/kentaroy47/vision-transformers-cifar10).

- To train a Swin-Transformer: 
```
git clone https://github.com/kentaroy47/vision-transformers-cifar10.git

python train_cifar10.py --net swin --n_epochs 400
```

- The pre-trained ```ResNet-50``` on ImageNet-1K follows [the Pytorch implmentation](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) with ```ResNet50_Weights.IMAGENET1K_V2```.

### Run Experiments
#### Train DAD on CIFAR-10
- Generate training data for MMD and denoiser:
```
cd dataset

python3 cifar10.py
```
- Train DAD:
```
# train DAD on WRN-28-10
python3 train.py --data 'CIFAR10' --model 'wrn28' --epochs 60 

# train DAD on WRN-70-16
python3 train.py --data 'CIFAR10' --model 'wrn70' --epochs 60 
```

#### Train DAD on ImageNet-1K
- Generate training data for MMD and denoiser:
```
cd dataset

python3 imagenet.py
```
- Generate adversarial data for training MMD-OPT and denoiser:

```
python3 adv_generator.py  --mode 'train' --data 'ImageNet' --model 'rn50' --attack 'mma'
```
- Train DAD:
```
python3 train.py --data 'ImageNet' --model 'rn50' --epochs 60 
```

#### Evaluate DAD against adaptive PGD+EOT whitebox attacks
```
python3 adaptive_whitebox_attack.py
```

#### Evaluate DAD against transfer attacks
- Generate transfer attacks:
```
# Take RN18 as an example

python3 adv_generator.py  --mode 'test' --data 'CIFAR10' --model 'rn18' --attack 'eotpgd' --epsilon 8/255

python3 adv_generator.py  --mode 'test' --data 'CIFAR10' --model 'rn18' --attack 'cw' --num-step 200 --epsilon 0.5

python3 adv_generator.py  --mode 'test' --data 'CIFAR10' --model 'rn18' --attack 'eotpgd' --epsilon 12/255

python3 adv_generator.py  --mode 'test' --data 'CIFAR10' --model 'rn18' --attack 'cw' --num-step 200 --epsilon 1.0
```

- Evaluate the performance against transfer attacks:
```
# Take RN18 as an example

python3 transfer_attack.py --model 'rn18' --epsilon 8/255

python3 transfer_attack.py --model 'rn18' --epsilon 12/255
```

### BPDA+EOT Implementation and Evaluation
The implementation and evaluation of BPDA+EOT strictly follows [the paper](https://arxiv.org/abs/2005.13525) with the following [GitHub](https://github.com/point0bar1/ebm-defense).


### License and Contributing
- This README is formatted based on [the NeurIPS guideline](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post any issues via GitHub.
