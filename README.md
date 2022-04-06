# Adversarial Detection without Model Information

This repository contains the code for the layer-wise energy separation training associated with the paper https://arxiv.org/pdf/2202.04271.pdf

## About the Work

Prior state-of-the-art adversarial detection works are classifier model dependent, i.e., they require classifier model outputs and parameters for training the detector or during adversarial detection. This makes their detection approach classifier model specific. Furthermore, classifier model outputs and parameters might not always be accessible. To this end, we propose a classifier model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the classifier model, with a layer-wise energy separation (LES) training to increase the separation between natural and adversarial energies. With this, we perform energy distribution-based adversarial detection. Our method achieves comparable performance with state-of-the-art detection works (ROC-AUC > 0.9) across a wide range of gradient, score and gaussian noise attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Furthermore, compared to prior works, our detection approach is light-weight, requires less amount of training data (40% of the actual dataset) and is transferable across different datasets.

<img src="/gifs/stage_1.gif" width="500" height="250"/>
<img src="/gifs/stage_2.gif" width="500" height="250"/>
<img src="/gifs/stage_3.gif" width="500" height="250"/>

## How to Setup the Environment
```shell
conda create -n env_name python=3.7
source activate env_name
pip install -r requirements.txt
```

## Generating the dataset
```shell
chmod +x generate_dataset.sh
./generate_dataset.sh
```

## Running the Layer-wise Energy Separation Training
```shell
chmod +x run_les_training.sh
./run_les_training.sh
```

## Citation

Please consider citing our paper:

```
@article{moitra2022adversarial,
  title={Adversarial Detection without Model Information},
  author={Moitra, Abhishek and Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2202.04271},
  year={2022}
}
```
More Documentation coming Soon. Until then, if you have any questions regarding code execution, please feel free to email me at abhishek.moitra@yale.edu
