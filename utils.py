"""
This is utility file containing all the helper functions required to run the 
model.
The idea for this file was somewhat inspired by the previous work done on BehaviouralCloning
https://github.com/ManajitPal/BehavioralCloning
"""

# First Let's import all the necessary libraries
import cv2
import matplotlib.image as mpimg
import os
import numpy as np
from sklearn.utils import shuffle

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
# Input Shape of our keras model
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess(image):
    # Crop the Image to remove Sky
    image = image[60:-25, :, :]
    # Resize Image to Match our Input Shape
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def load_image(data_dir, image_file):
    image_file = image_file.split("\\")[-1]
    return mpimg.imread(os.path.join(data_dir,"IMG",image_file.strip()))


def choose_image(data_dir, center, left, right, steering_angle):
    """
    We have Three sets of Images in our dataset. For Training purpose we are goin
    to randomly choose one type of image and will adjust the steering angle according to that.
    """
    rand = np.random.choice(3)
    if rand == 0:
        return load_image(data_dir, left), steering_angle + 0.25
    elif rand == 1:
        return load_image(data_dir, right), steering_angle - 0.25
    else:
        return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    if np.random.rand() < 0.6:
        image = cv2.flip(image, 1)
        steering_angle = - steering_angle
    return image, steering_angle


def brighten_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    default_bias = 0.25
    brightness = default_bias + np.random.uniform()    
    if np.random.rand() < 0.6:
        hsv_image[:,:, 2] = hsv_image[:,:,2]*brightness
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def augment(data_dir, center, left, right, steering_angle):
    image, steering_angle = choose_image(
        data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image = brighten_image(image)
    return image, steering_angle

# Now it's time to write the final function for this file A.K.A. Our batch generator
# function that will generate training data parallely using CPU while our model is
# training on GPU


def batch_generator(data_dir, images_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(images_paths.shape[0]):
            center, left, right = images_paths[index]
            steering_angle = steering_angles[index]

            if is_training and np.random.rand() < 0.7:
                image, steering_angle = augment(
                    data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield shuffle(images, steers)
