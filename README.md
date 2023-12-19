# AAAI24-DePRL

Here are the key components and instructions for implementing the algorithms under the PFL (Personalized Federated Learning) scenario, as described in our paper.

# Usage

1. Prerequisites: Ubuntu 20.04, Python v3.5+, PyTorch, and CUDA environment
2. The "./Main.py" file contains configuration settings and the basic framework for Federated Learning.
3. The "./Sims.py" file describes simulators for clients and the central server.
4. The "./Utils.py" file includes necessary functions and provides guidance on obtaining training and testing data.
5. The "./Settings.py" file specifies the required packages and settings.
6. The "./Models" folder contains code implementations for DNN, AlexNet, VGG-11, and ResNet-18 models.

# Implementation

1. To execute the algorithms, run the "./Main.py" file using the following command: ``python3 ./Main.py''
2. Adjust the parameters and configurations within the ``./Main.py'' file to suit your specific needs.

# Citation
If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2023defl,
  title={DePRL: Achieving Linear Convergence Speedup in Personalized Decentralized Learning with Shared Representations},
  author={Guojun Xiong, Gang Yan, Shiqiang Wang, Jian Li},
  booktitle={Proc. of AAAI},
  year={2024}
}
```
