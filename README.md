# AU-DFER
source code for AU-enhanced DFER
## for M3DFEL
### Libraries and Dependencies
### Data Preparation
move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to [datasets](M3DFEL_AU/datasets).
### Change Directory and Settings
All settings are included in [options.py](M3DFEL_AU/options.py). Please change the parameter root into the root directory of datasets, and other parameters if necessary.
### run the training code
```bash
python main.py
```
## for MAE-DFER
Please pretrain a model on voxleb2 or download a pretrained model first. This program is only for finetuning.
### Libraries and Dependencies
### Data Preparation
move AU_DFEW.txt to [saved/data/dfew](MAE-DFER_AU/saved/data/dfew), or or AU_FERV39K.txt to [saved/data/ferv39k](MAE-DFER_AU/saved/data/ferv39k).
After adding the AU label txt, edit data_path in [dfew.py](MAE-DFER_AU/preprocess/dfew.py), or [ferv39k.py](MAE-DFER_AU/preprocess/ferv39k.py).
Then run the preprocessing code.
```bash
cd ./MAE-DFER_AU/preprocess
python dfew.py
python ferv39k.py
```
### Change Settings
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
### Data Preparation
Please move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to [annotation](Former-DFER_AU/annotation)
Then run the annotation code.
```bash
cd ./Former-DFER_AU/annotation
python script.py
```
If you are using DFEW, please run following AU label set split:
```bash
python gen_sets_AU.py
```
### Run training code
For 5-sets traning on DFEW:
```bash
cd ..
sh sh main_DFEW_trainer.sh
```
For FERV39K:
```bash
cd ..
python main_FERV39K.py
```
