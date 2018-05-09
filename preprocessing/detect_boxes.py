from PIL import Image, ImageChops
from glob import glob
from os import path, walk
import imutils
import cv2

# local folder containing image dataset
dataset = '/home/sevvy/Documents/owe10/Eindopdracht/kvasir-dataset/'
# retrieve all subfolders containing the images per class as a list
kvasir_classes = next(walk(dataset))[1]


def kvasir_set():
    """
    Read images from a local folder directory and filter
    on images not containing a green area (not suitable for processing with CNN)
    :return: dictionary with key: classname and value: list of image data (3D arrays)
    """
    kvasir_dict = {}
    # iterate folder (class) names
    for k_class in kvasir_classes:
        # get full path of subfolder
        class_path = path.join(dataset, k_class)
        # initialize dictionary with class name as key
        kvasir_dict[k_class] = []

        # iterate images in subfolder
        for impath in glob(path.join(class_path, '*.jpg')):
            # load the image and resize it to a smaller factor so that
            # the shapes can be approximated better
            image = cv2.imread(impath)
            resized = imutils.resize(image, width=300)

            # blur the image slightly, then convert it to grayscale and
            # consecutively to the L*a*b* color spaces for better thresholding
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
            # get binary image with thresholding
            thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
            # convert to PIL image
            pilthresh = Image.fromarray(thresh)
            # invert colors and check if image is all white (if green area is present
            # this area will not be removed by thresholding)
            # if image is all white (does not contain green box), add to dict for further processing
            if not ImageChops.invert(pilthresh).getbbox():
                kvasir_dict[k_class].append(impath)

    return kvasir_dict


