# AU-DFER
source code for AU-enhanced DFER
## Program Preview
```
.
├── AU_DFEW.txt
├── AU_FERV39K.txt
├── Former-DFER_AU
│   ├── annotation
│   │   ├── DFEW_set_1_test.txt
│   │   ├── DFEW_set_1_train.txt
│   │   ├── DFEW_set_2_test.txt
│   │   ├── DFEW_set_2_train.txt
│   │   ├── DFEW_set_3_test.txt
│   │   ├── DFEW_set_3_train.txt
│   │   ├── DFEW_set_4_test.txt
│   │   ├── DFEW_set_4_train.txt
│   │   ├── DFEW_set_5_test.txt
│   │   ├── DFEW_set_5_train.txt
│   │   ├── FERV39K_test.txt
│   │   ├── FERV39K_train.txt
│   │   ├── gen_sets_AU.py
│   │   └── script.py
│   ├── dataloader
│   │   ├── dataset_AFEW.py
│   │   ├── dataset_DFEW.py
│   │   ├── dataset_FERV39K.py
│   │   └── video_transform.py
│   ├── main_DFEW.py
│   ├── main_DFEW_trainer.sh
│   ├── main_FERV39K.py
│   ├── models
│   │   ├── ST_Former.py
│   │   ├── S_Former.py
│   │   └── T_Former.py
│   └── requirements.txt
├── LICENSE
├── M3DFEL_AU
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── dataset_DFEW_AU_weighted.py
│   │   ├── dataset_FERV39K_AU_weighted.py
│   │   └── video_transform.py
│   ├── main.py
│   ├── models
│   │   ├── M3DFEL_with_AU_weighted.py
│   │   └── __init__.py
│   ├── options.py
│   ├── requirements.txt
│   ├── solver_with_AU_weighted.py
│   └── utils.py
├── MAE-DFER_AU
│   ├── datasets.py
│   ├── engine_for_finetuning.py
│   ├── engine_for_pretraining.py
│   ├── environment.yml
│   ├── functional.py
│   ├── kinetics.py
│   ├── preprocess
│   │   ├── dfew.py
│   │   ├── ferv39k.py
│   │   └── voxceleb2.py
│   ├── requirements.txt
│   ├── saved
│   │   └── data
│   │       ├── dfew
│   │       │   └── org
│   │       │       ├── split01
│   │       │       │   ├── test.csv
│   │       │       │   ├── train.csv
│   │       │       │   └── val.csv
│   │       │       ├── split02
│   │       │       │   ├── test.csv
│   │       │       │   ├── train.csv
│   │       │       │   └── val.csv
│   │       │       ├── split03
│   │       │       │   ├── test.csv
│   │       │       │   ├── train.csv
│   │       │       │   └── val.csv
│   │       │       ├── split04
│   │       │       │   ├── test.csv
│   │       │       │   ├── train.csv
│   │       │       │   └── val.csv
│   │       │       └── split05
│   │       │           ├── test.csv
│   │       │           ├── train.csv
│   │       │           └── val.csv
│   │       └── ferv39k
│   │           └── all_scenes
│   │               ├── test.csv
│   │               └── train.csv
│   └── scripts
│       ├── dfew
│       │   └── finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh
│       ├── ferv39k
│       │   └── finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh
│       └── voxceleb2
│           └── pretrain_local_global_attn_depth16_region_size2510_with_diff_target_102.sh
├── README.md
├── directory_tree.md
└── log
    ├── Former-DFER_DFEW
    │   ├── Former-DFER_AU
    │   │   ├── set1-log.txt
    │   │   ├── set2-log.txt
    │   │   ├── set3-log.txt
    │   │   ├── set4-log.txt
    │   │   └── set5-log.txt
    │   └── Former-DFER_baseline
    │       ├── set1-log.txt
    │       ├── set2-log.txt
    │       ├── set3-log.txt
    │       ├── set4-log.txt
    │       └── set5-log.txt
    ├── Former-DFER_FERV39K
    │   ├── FormerDFER_AU.txt
    │   └── FormerDFER_baseline.txt
    ├── M3DFEL_DFEW
    │   ├── M3DFEL_AU.txt
    │   └── M3DFEL_baseline.txt
    ├── M3DFEL_FERV39K
    │   ├── M3DFEL_AU.txt
    │   └── M3DFEL_baseline.txt
    ├── MAE-DFER_DFEW
    │   ├── MAE-DFER_AU.txt
    │   └── MAE-DFER_baseline.txt
    └── MAE-DFER_FERV39K
        ├── MAE-DFER-baseline.txt
        └── MAE-DFER_AU.txt
```


## Specification of Key Arguments
* `--AU_ratio`: the ratio of AU loss and expression loss, a float between 0 and 1.
* `--posw_option`: determines the method of pos_weight for AU loss calculation, 'global', 'distinct', 'minor' accepted.
## AU Label download
If you don't want to run AU detection model to obtain AU label, you can download via the following link:
https://drive.google.com/drive/folders/1MY8hO9eCHHuJb0DcmRG8ALE7k-csW01y?usp=sharing
## for M3DFEL
### Libraries and Dependencies
python 3.9 is required. For dependencies, please refer to [requirements.txt](M3DFEL_AU/requirements.txt).
### Data Preparation
move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to [datasets](M3DFEL_AU/datasets).
### Choose parameters
All key parameters are included in [options.py](M3DFEL_AU/options.py). Please change the parameter `--root` into the root directory of datasets, and other parameters if necessary.
### run the training code
```bash
python main.py
```
## for MAE-DFER
Please pretrain a model on voxleb2 or download a pretrained model first. This program is only for finetuning.
### Libraries and Dependencies
python 3.8 is required. For dependencies, please refer to [requirements.txt](MAE-DFER_AU/requirements.txt).
### Data Preparation
move AU_DFEW.txt to [saved/data/dfew](MAE-DFER_AU/saved/data/dfew), or or AU_FERV39K.txt to [saved/data/ferv39k](MAE-DFER_AU/saved/data/ferv39k).
After adding the AU label txt, edit data_path in [dfew.py](MAE-DFER_AU/preprocess/dfew.py), or [ferv39k.py](MAE-DFER_AU/preprocess/ferv39k.py).
Then run the preprocessing code:
```bash
cd ./MAE-DFER_AU/preprocess
python dfew.py
python ferv39k.py
```
### Choose parameters
AU_ratio and posw_settings can be changed in [finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh](MAE-DFER_AU/scripts/dfew/finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh) or [finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh](MAE-DFER_AU/scripts/ferv39k/finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh).
You can also change other parameters if you wish.

### Run fine-tuning code
If you are fine-tuning on DFEW:
```bash
sh scripts/dfew/finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh
```
If you are fine-tuning on FERV39K:
```bash
sh scripts/ferv39k/finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh
```
## for Former-DFER
### Libraries and Dependencies
python 3.9 is required. For dependencies, please refer to [requirements.txt](Former-DFER_AU/requirements.txt).
### Data Preparation
Please move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to [annotation](Former-DFER_AU/annotation).
Then run the annotation code.
```bash
cd ./Former-DFER_AU/annotation
python script.py
```
If you are using DFEW, please run following AU label set split:
```bash
python gen_sets_AU.py
```
### Choose parameters
All key parameters are specified in [main_DFEW.py](Former-DFER_AU/main_DFEW.py) and [main_FERV39K.py](Former-DFER_AU/main_FERV39K.py).
### Run training code
For 5-sets trainng on DFEW:
```bash
cd ..
sh main_DFEW_trainer.sh
```
For FERV39K:
```bash
cd ..
python main_FERV39K.py
```
