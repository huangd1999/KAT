# KAT: Towards Building More Robust Models with Kolmogorov–Arnold Networks-Based Adversarial Training
  
## Paper Abstract
Deep Neural Networks (DNNs) have demonstrated remarkable performance in various machine learning tasks. However, their vulnerability to adversarial attacks raises concerns about their robustness and reliability in real-world applications. Inspired by recent works that utilize Kolmogorov–Arnold Networks (KAN) to replace Multi Layer Perceptron to further improve DNNs effectiveness, in this paper, we propose a novel approach to improve the robustness of DNNs by leveraging the mathematical properties of KAN in adversarial training. Our method, named KAT (Kolmogorov-Arnold Network-based Adversarial Training), aims to replace traditional MLP in DNNs with KANs for enhanced resilience against adversarial attacks. Through extensive experiments, we demonstrate that our KAT approach significantly improves the robustness of DNNs under various adversarial attacks, including PGD and AutoAttack. Our results demonstrate that the PGD-50 and AutoAttack accuracy for PAT increased from 49.82% and 47.42% to 51.78% and 48.39% compared with AT, respectively. Our experiment results show that KAN can further improve the robustness of DNNs compared with MLP.


## Usage
**Train a ResNet18 model on CIFAR10:**

*python train.py --data_root dataset --dataset CIFAR10 --weight_decay 3.5e-3 --lr 0.01 --batch_size 128 --epoch 120 --model resnet18*

**Train a WRN-34-10 model on CIFAR10:**

*python train.py --data_root dataset --dataset CIFAR10 --weight_decay 5e-4 --lr 0.1 --batch_size 128 --epoch 60 --model wrn-34-10*

**Test a model under PGD-50 attack:**

*python test.py --weights checkpoint --attack PGD --step 50 --dataset dataset_name --model model_name*

## Citation
If you find our work useful in your research, please consider citing:
````
@InProceedings{Bu_2023_ICCV,
    author    = {Bu, Qingwen and Huang, Dong and Cui, Heming},
    title     = {Towards Building More Robust Models with Frequency Bias},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4402-4411}
}
````
