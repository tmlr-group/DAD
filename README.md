# Code implementations for NeurIPS 2024 submission #6999
## DAD: Towards Robust Classification via Distributional-discrepancy-based Adversarial Defense

 #### Abstract
 Statistical adversarial data detection (SADD) identifies whether an upcoming batch contains adversarial examples (AEs). Given the distributional discrepancies between natural examples (NEs) and AEs, SADD offers statistical guarantees against adversarial attacks. However, an undeniable problem of SADD-based methods is that they will discard data batches that contain AEs, leading to poor model deployment during testing time. In this paper, to solve this problem, we propose a new statistical adversarial defence method, namely Distributional-discrepancy-based Adversarial Defense (DAD). In the training phase, DAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT and then trains a denoiser by minimizing the MMD-OPT between NEs and AEs. In the testing phase, DAD uses MMD-OPT to detect AEs and then denoises them instead of discarding them. Theoretically, we justify that narrowing the distributional discrepancy can help reduce the upper bound of risk on AEs. Empirically, DAD will not discard data compared to detection-based defense methods and outperforms current state-of-the-art adversarial training (AT) and adversarial purification (AP) methods by notably improving natural and robust accuracy simultaneously on CIFAR-10 and ImageNet-1K against various adversarial attacks, including adaptive attacks, which pioneers a new road for statistical adversarial defense methods.

#### Figure 1: The illustration of our method.
![pipeline](images/pipeline.jpg)

### Requirement
- This codebase is written for ```python3``` and ```pytorch```.
- To install necessay python packages, run ```pip install -r requirements.txt```.

### AutoAttack Implementation
The implementation of AutoAttack used in our paper can be found in [RobustBench](https://robustbench.github.io/): [Github Link](https://github.com/RobustBench/robustbench).

To use AutoAttack:

```
git clone https://github.com/fra31/auto-attack.git
```

