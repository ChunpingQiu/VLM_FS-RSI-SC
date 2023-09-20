# Setup
...Moreover, we proposed a pipeline that harnesses the capabilities of powerful large vision-language models (VLMs) as image encoders, establishing new baselines for FS-RSI-SC on commonly used datasets under standard experimental settings. 

Please follow the official setup of CLIP here: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
Please follow the official setup of BLIP here: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)

# Folder Structure
  ```
  pytorch-template/
  ├── run_classifier_fromFeatures.sh - the main file to run classifier_fromFeatures.py
  ├── saveFeatures_subfolder_aid_nwpu_whu.py - extract features from images and save
  ├── classifier_fromFeatures.py
  │
  │
  ├── nwpu-resisc45/ - extracted features, class-wise split following the common setup
  │   ├── train - train split of the common setup
  │   ├── test - 
  │   ├── val - 
  │   
  ├── acc.txt - evaluation results
 ```     

# Data

To reproduce the results in Table 4, please download datasets.
The train-val-test split is following a common one in litearature, and the training set is not actually used for training models.

First extract feautres using saveFeatures_subfolder_aid_nwpu_whu.py, and then set the path in run_classifier_fromFeatures.sh, and  run: ```bash run_classifier_fromFeatures.sh``` and results will be saved in the folder "acc.txt".

# Cite:

If you use this code for your research, please cite:

```
```