# Preprocessing module for doing necessary preperation on images before feeding to a machine learning algorithm

# import modules from local package 'preprocessing'
from preprocessing import detect_boxes as db
from preprocessing import text_vanish as tv
from preprocessing import crop_edges as ce

import cv2
import os

# base directory for output images
base_dir = '/home/sevvy/Documents/owe10/Eindopdracht/processed'

# get a dataset of the filtered images (as a dictionary)
kvasir_doc = db.kvasir_set()

# iterate classes from the image dataset
for class_name in kvasir_doc:
    # define output directory with name of the class
    out_dir = os.path.join(base_dir, class_name)
   # get the list of images for processing
    imlist = kvasir_doc[class_name]

    # iterate the images in the list
    for impath in imlist:
        # attempt to remove text from image (if present)
        processed = tv.Vanisher(impath).do_vanish()
        # crop black edges from image
        cropped = ce.Crop(processed).do_crop()
        # final image is resized to one uniform size
        final = cv2.resize(cropped, (625, 532))

        # difine filename with suffix and write to output directory
        filename = os.path.basename(impath).strip('.png') + '_prsd'
        out_path = os.path.join(out_dir, filename + '.png')
        cv2.imwrite(out_path, final)
