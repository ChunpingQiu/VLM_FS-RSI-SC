# image features are first extracted (using saveFeatures_subfolder_aid_nwpu_whu.py) and saved into the path: --root ./
CUDA_VISIBLE_DEVICES=2 python classifier_fromFeatures.py  --root ./ --dataset nwpu-resisc45 --way 45 --sets train val test
