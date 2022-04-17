import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm


import os, os.path

import logging


from utils import get_classes, prepare_base_folder, prepare_classes_folders, prepare_video_folders
from utils import merge_u_v_into_image

path = "../../datasets/ucf101_tvl1_flow/tvl1_flow/"


def get_number_frames_in_clip(route):
    #print(f"La longitud es {len(os.listdir(route+'u/'+name))}")
    #print(f"La ruta es {route+'u/'+name}")
    #input("Pauss")
    return len(next(os.walk(route))[2])

def move_images(route,destination, class_name, split):
    

    n_frames = get_number_frames_in_clip(route)

    

    for frame in range(1, n_frames):
        #print(f"Estamos en el frame {frame}")
        #input("Pausa")
        
        pre_img = Image.open(route + f"0000" + str(frame-1) + ".jpg" ) 
        post_img = Image.open(route + f"0000" + str(frame) + ".jpg" )

        img_res = ImageChops.subtract(post_img, pre_img, 1 , 0)
        frame_out = frame-1
        img_res.save(destination+ f"{frame_out:06d}.jpg")




def prepare(split_id = 1, route = "", destination = ""):
    """
        split_id: Identifier of the split selected (1, 2 or  3)
        route: Where the flow
        destination: Where the flow will go
    """
    
    assert route != ""

    classes = get_classes("../../dataloaders/labels/kth_labels.txt")

    prepare_base_folder(destination)
    for class_i in (pbar := tqdm(classes)):
        pbar.set_description("Processing %s" % class_i)
        prepare_classes_folders(destination, class_i)

        for split in ["train", "test", "val"]:
        
            files = next(os.walk(route+split+"/"+class_i+"/"))[1]

            for video_name in (sbar := tqdm(files, leave=False)):
                sbar.set_description("Processing %s" % video_name[:20])

                os.mkdir(destination + split+"/"+class_i+"/"+video_name)
                move_images(
                    route+split+"/"+class_i+"/"+video_name +"/",
                    destination+split+"/"+class_i+"/"+video_name +"/",
                    class_i,
                    split
                )
        

if __name__ == '__main__':
    prepare(
        split_id="1",
        route="../../datasets/KTH/Processed/",
        destination="../../datasets/KTH_RGB_Diff/")