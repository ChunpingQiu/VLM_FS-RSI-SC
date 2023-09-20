import torch
print("Torch version:", torch.__version__)

from PIL import Image
from tqdm import tqdm
import numpy as np
import os

from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import argparse

import random
from scipy.special import softmax

# conda activate /media/admin1/envs/anaconda3/envs/lengEnv

def get_args_parser():

    parser = argparse.ArgumentParser('few-shot image scene classification', add_help=False)

    parser.add_argument('--sets', nargs='+', help="List of split", default=['test'])                         

    parser.add_argument('--root', default="/root/data1/image_scene_classification/blip_exp/features/", type=str,
                        help='path to exttracted features')
    
    parser.add_argument('--dataset', default='ucm', type=str,
                        help='data to test')

    parser.add_argument('--way', default=5, type=int,
                        help='number of classes')

    args = parser.parse_args()

    return args

#calculate prototypes using support sets
def fewshot_classifier(features_per_class):

    fewshot_weights = []

    for features in features_per_class:
        
        class_embedding = features.mean(axis=0)

        class_embedding /= np.linalg.norm(class_embedding)
        
        fewshot_weights.append(class_embedding)      

    fewshot_weights = np.stack(fewshot_weights, axis=1)

    return fewshot_weights


def main(args):

    if args.dataset == 'ucm':
        num_per_class = 100
        cls_num = 21
    
    if args.dataset == 'nwpu-resisc45':
        num_per_class = 700
        cls_num = 45
        
    # repeat and avearage
    repeat_num = 50
    
    way = args.way
    
    # the sets to be evaluated, the train/val set is not used for training, the split is just following the original naming
    # sets = ['train', 'val', 'test']#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # sets = ['test']#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sets = args.sets
    ####################################

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    ##############################################read features
    all_features = []
    classes_all = []
    
    for split in sets:
        
        folder = args.root  + args.dataset + "/" + split + "/"
        classes = os.listdir(folder)
        classes=sorted(classes)
        print(classes)
        classes_all = classes_all + classes
    
        for cls in classes:
            # print(cls)
            image_features_class = []
            directory_path = folder + cls + "/"
            # Loop through each file in the directory
            for filename in os.listdir(directory_path): 
                file = os.path.join(directory_path, filename)
                feature = np.load(file)
                image_features_class.append(feature)

            image_features_class = np.stack(image_features_class, axis=0)
            # print(image_features_class.shape)(100, 1, 256)
            
            image_features_class = np.squeeze(image_features_class) 
            # print(image_features_class.shape)

            all_features.append(image_features_class)

    # print(len(all_features))#21
    # print(classes_all)
    
    if sets == ['train', 'val', 'test']:
        assert cls_num == len(classes_all)
    cls_num_test = len(classes_all)
    
    print('way number', way)
    print('split set', sets)
    ############################################## N shot X way
    
    accuracy = np.zeros((2, repeat_num))
    row = 0
    for shot in [1, 5]:
        
        # repeat and avearage, if all classes are used for testing, each iteration outputs the same results
        for iter in np.arange(repeat_num):
            
            #way
            # select the novel class with the size of way
            numbers = list(range(0, cls_num_test))
            random.shuffle(numbers)
            # Select the first way classes from the shuffled list
            classes_test_idx = numbers[:way]
            # print(classes_test_idx)         
            
            #shot
            # Generate a list of numbers from 1 to 100
            numbers = list(range(0, num_per_class))
            # Shuffle the list
            random.shuffle(numbers)
            # Select the first N numbers from the shuffled list, for the support set
            N = numbers[:shot]
            # Print the randomly chosen numbers
            # print(N)

            #obtain the support and query set
            support_set = []
            query_set = []
            for cls_idx in classes_test_idx:#np.arange(len(classes_test)):

                support = all_features[cls_idx][N,:]
                support_set.append(support)

                #all queries
                query_set.append(np.delete(all_features[cls_idx], N, axis=0))

            # print(query_set[0].shape)(99, 256)
            query_set  = np.row_stack(query_set)
            # print(query_set.shape)#(2079, 256)
            #labels
            targets = np.repeat(np.arange(way), num_per_class - len(N))

            # calculate the protypes using support set
            fewshot_weights = fewshot_classifier(support_set)

            # print(targets.shape, fewshot_weights.shape)#(2079,) (256, 21)

            #calculate similarity
            logits_image_image = 100. * query_set @ fewshot_weights
            probs_0 = softmax(logits_image_image, axis=-1)
            outputAll = np.uint8(np.argmax(probs_0, axis=1))
            
            # print(outputAll.shape)
            # calculate accuracy
            accuracy[row, iter] = accuracy_score(targets, outputAll)
            # print(accuracy)
            
        row = row + 1
 
    # print(accuracy)
    # Calculate the mean and standard deviation for each row
    row_means = np.mean(accuracy, axis=1)*100
    row_stds = np.std(accuracy, axis=1)
    
    # print(row_means,row_stds)
    output = f"{way}, {row_means[0]:.2f} ± {row_stds[0]:.2f},  {row_means[1]:.2f} ± {row_stds[1]:.2f}, {args.root}\n"
    print(args.root, output)
    
    # Write the data to the file
    with open('./acc.txt', 'a') as f:
        f.write(output)

if __name__ == '__main__':

    args = get_args_parser()

    main(args)