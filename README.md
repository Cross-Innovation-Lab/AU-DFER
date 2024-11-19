# AU-DFER
source code for AU-enhanced DFER
## for M3DFEL
### Data Preparation
move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to [datasets](M3DFEL_AU/datasets).
### Libraries and Dependencies
### Change Directory and Settings
All settings are included in [options.py](M3DFEL/options.py). Please change the parameter root into the root directory of datasets, and other parameters if necessary.
### run the training code
```bash
python main.py
```
## for MAE-DFER
### Data Preparation
move AU_DFEW.txt or AU_FERV39K.txt (according to the dataset for trainning) to ./datasets
