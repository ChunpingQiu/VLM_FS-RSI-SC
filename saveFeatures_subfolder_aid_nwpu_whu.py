import torch
print("Torch version:", torch.__version__)

from PIL import Image
# from tqdm import tqdm
import numpy as np
import os

import argparse

import pathlib 

from lavis.models import load_model_and_preprocess
import clip

# conda activate XXX
# CUDA_VISIBLE_DEVICES=1 python saveFeatures_subfolder_aid_nwpu_whu.py --model_source clip --model_name RN101 --dataset nwpu-resisc45 
# nwpu-resisc45 whu19 ucm aid

# extract features for each image in the (train val test) subdolders 
def get_args_parser():

    parser = argparse.ArgumentParser('metric learning for image scene classification', add_help=False)

    parser.add_argument('--root', default="./", type=str,
                        help='path to data')
    parser.add_argument('--dataset', default='ucm', type=str,
                        help='dataset name')
    parser.add_argument('--model_source', default='blip', type=str, metavar='MODEL',
                        help='Name of model source to train')      
    parser.add_argument('--model_name', default='RN50', type=str, metavar='MODEL',
                        help='Name of model clip')     
    # print(clip.available_models())
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] 

    ##############################

    args = parser.parse_args()

    return args

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset=args.dataset

    root=args.root
    
    for split in ['train', 'val', 'test']:
        
        folder=root+dataset+'/'+split#
        
        classes = os.listdir(folder)
        classes=sorted(classes)
        print(classes)

        if args.model_source == 'blip':
            model, preprocess, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)#args.model_name
            model_name_save='blip2_feature_extractor'
        
        if args.model_source == 'clip':
            model, preprocess = clip.load(args.model_name, device=device)
            model_name_save=args.model_name
                
        ##############################################
        with torch.no_grad():

            for cls in classes:
                print(cls)

                directory_path = folder + "/" + cls + "/"
                
                output_dir_image = root+args.model_source+'_exp/features/'+model_name_save+'/'+dataset+'/'+split+'/'+ cls + "/"
                pathlib.Path(output_dir_image).mkdir(parents=True, exist_ok=True)
        
        
                # Loop through each file in the directory
                for filename in os.listdir(directory_path):

                    # print(output_dir_image+filename[:-3]+'_image_features.npy')      
                        
                    img = Image.open(os.path.join(directory_path, filename))
                    # print(img)
                    # image = preprocess["eval"](img)
                    # image = image[np.newaxis, ...]
                    # print(image.shape)

                    if args.model_source == 'blip':
                        image = preprocess["eval"](img)
                        image = image[np.newaxis, ...]
                        image_features = model.extract_features( {"image": image.to(device), "text_input": []}, mode="image").image_embeds_proj[:, 0]          


                    if args.model_source == 'clip':
                        image = preprocess(img)
                        image = image[np.newaxis, ...]
                        # print(image.shape)
                        # image_features = model.extract_features( {"image": image.to(device), "text_input": []}, mode="image").image_embeds_proj[:, 0]   
                        image_features = model.encode_image(image.to(device)) 
                        
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    #################################################save features ucm_features  output_dir_caption
                    # print(image_features.shape, imag_captione_features.shape)           
                    np.save(output_dir_image+filename[:-3]+'npy', image_features.cpu().numpy())


if __name__ == '__main__':

    args = get_args_parser()

    main(args)