# Cross-Domain Few-Shot Learning (CD-FSL) Benchmark

### Website
#### https://www.learning-with-limited-labels.com/

### Paper

Please cite the following paper in use of this evaluation framework: https://arxiv.org/pdf/1912.07200.pdf

```
@article{guo2019new,
  title={A New Benchmark for Evaluation of Cross-Domain Few-Shot Learning},
  author={Guo, Yunhui and Codella, Noel CF and Karlinsky, Leonid and Smith, John R and Rosing, Tajana and Feris, Rogerio},
  journal={arXiv preprint arXiv:1912.07200},
  year={2019}
}
```

## Introduction

The Cross-Domain Few-Shot Learning (CD-FSL) challenge benchmark includes data from the CropDiseases [1], EuroSAT [2], ISIC2018 [3-4], and ChestX [5] datasets, which covers plant disease images, satellite images, dermoscopic images of skin lesions, and X-ray images, respectively. The selected datasets reflect real-world use cases for few-shot learning since collecting enough examples from above domains is often difficult, expensive, or in some cases not possible. In addition, they demonstrate the following spectrum of readily quantifiable domain shifts from ImageNet: 1) CropDiseases images are most similar as they include perspective color images of natural elements, but are more specialized than anything available in ImageNet, 2) EuroSAT images are less similar as they have lost perspective distortion, but are still color images of natural scenes, 3) ISIC2018 images are even less similar as they have lost perspective distortion and no longer represent natural scenes, and 4) ChestX images are the most dissimilar as they have lost perspective distortion, all color, and do not represent natural scenes.

## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet

### Target domains: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.kitware.com/#phase/5abcbc6f56357d0139260e66

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

## General information

* **No meta-learning in-domain**
* Only ImageNet based models or meta-learning allowed.
* 5-way classification
* n-shot, for varying n per dataset
* 600 randomly selected few-shot 5-way trials up to 50-shot (scripts provided to generate the trials)
* Average accuracy across all trials reported for evaluation.

* **For generating the trials for evaluation, please refer to finetune.py and the examples below**

## Specific Tasks:

**EuroSAT**

  • Shots: n = {5, 20, 50}

**ISIC2018**

  • Shots: n = {5, 20, 50}

**Plant Disease**

  • Shots: n = {5, 20, 50}

**ChestX-Ray8**

  • Shots: n = {5, 20, 50}


## Enviroment

Python 3.5.5

Pytorch 0.4.1

h5py 2.9.0

## Steps

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links.

2. Download miniImageNet. 

```bash
    Change directory to ./filelists/miniImagenet
    run source ./download_miniImagenet.sh
```

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to (i.e. the `imagenet_path` variable should point directly to the training partition in the `train` subfolder). You can skip the first two steps if these datasets are already downloaded.

4. Run miniImageNet training configuration. 

```bash
    Change directory to ./filelists/miniImagenet
    run source ./configure_miniImagenet.sh
```

5. Train base models on miniImageNet

    • *Standard supervised learning on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug
    ```

    • *Train meta-learning method (protonet) on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method protonet --n_shot 5 --train_aug
    ```

6. Save features for evaluation (optional, if there is no need to adapt the features during testing) 

    • *Save features for testing*

    ```bash
        python save_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

7. Test with saved features (optional, if there is no need to adapt the features during testing) 

    ```bash
        python test_with_saved_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

8. Test

    • *Finetune with frozen model backbone*: 
 
    ```bash
        python finetune.py --model ResNet10 --method baseline  --train_aug --n_shot 5 --freeze_backbone
    ```

    • *Finetune*

    ```bash
        python finetune.py --model ResNet10 --method baseline  --train_aug --n_shot 5 
    ```
    
    Output: 600 Test Acc = 49.91% +- 0.44%

9. For testing your own methods, simply replace the function **finetune()** in `finetune.py` with your own method. Your method should at least have the following arguments,

    • *novel_loader: data loader for the corresponding dataset (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8)*

    • *n_query: number of query images per class*

    • *n_way: number of shots*

    • *n_support: number of support images per class*

## References

[1] Sharada P Mohanty, David P Hughes, and Marcel Salathe. Using deep learning for image
based plant disease detection. Frontiers in plant science, 7:1419, 2016

[2] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel
dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of
Selected Topics in Applied Earth Observations and Remote Sensing , 12(7):2217–2226, 2019.

[3] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific data, 5:180161, 2018.

[4] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen
Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael
Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge
hosted by the international skin imaging collaboration (isic). arXiv preprint.
arXiv:1902.03368, 2019

[5] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald
M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly
supervised classification and localization of common thorax diseases. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 2097–2106, 2017

