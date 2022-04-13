import os
import numpy as np
from PIL import Image

def get_classes(labels_file = "../../dataloaders/labels/hmdb_labels.txt", skip_first_n = 0):
    """
        Opens label file and returns a label list.

        Labels file have a structure of "[label_id] [label_name]" per line.

        Parameters:
            - labels_file (str): Location of the label file
    """
    
    labels = []

    with open(labels_file, "r") as file:
        for i, line in enumerate(file):
            if i < skip_first_n:
                continue
            labels += [line.split()[1]]

    return labels

def prepare_base_folder(destination ):
    for i in ["train", "test", "val"]:
        if not os.path.exists(destination + i):
            os.mkdir(destination + i)

def prepare_classes_folders(destination, class_name):
    for i in ["train", "test", "val"]:
        if not os.path.exists(destination + i+ "/" + class_name):
            os.mkdir(destination + i + "/" + class_name)

def prepare_video_folders(destination, class_name, video_name, split):
    if not os.path.exists(destination + split+ "/" + class_name + "/" + video_name):
        os.mkdir(destination + split + "/" + class_name + "/" + video_name)


# Image Utils

def merge_u_v_into_image(path, file):
    """
        Merges the u and v component from grayscales to a image with 3 channels.

        path = ../datasets/hmdb51_tvl1_flow/tvl1_flow/
        file = #20_Rhythm_clap_u_nm_np1_fr_goo_0/frame000055.jpg
    """

    u_img = Image.open(path + "u/" + file).convert('RGB') 
    v_img = Image.open(path + "v/" + file).convert('RGB') 

    r_u, _, _ = u_img.split()
    _, g_v, b = v_img.split()

    b = b.point(lambda i: 127) # Average color is 127

    merged =  Image.merge('RGB', (r_u, g_v, b))

    resized = merged.resize((171,128), Image.ANTIALIAS)

    return resized