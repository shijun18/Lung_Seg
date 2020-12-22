import numpy as np
import torch
from PIL import Image
import random


class RandomErase2D(object):
    '''
    Data augmentation method.
    Args:

    '''
    def __init__(self, window_size=(64,64), scale_flag=True):
        self.window_size = window_size
        self.scale_flag = scale_flag
    
    def __call__(self, sample):
        if self.scale_flag:
            h_factor = np.random.uniform(0.5, 1)
            w_factor = np.random.uniform(0.5, 1)
            max_h, max_w = np.uint8(self.window_size[0]*h_factor),np.uint8(self.window_size[1]*w_factor)
        else:
            max_h, max_w = self.window_size
        image = sample['image']
        mask = sample['mask']

        h,w = image.shape
        roi_window = []
        
        if np.sum(mask) !=0:
            roi_nz = np.nonzero(mask)
            roi_window.append((
                np.maximum((np.amin(roi_nz[0]) - max_h//2), 0),
                np.minimum((np.amax(roi_nz[0]) + max_h//2), h)
            ))

            roi_window.append((
                np.maximum((np.amin(roi_nz[1]) - max_w//2), 0),
                np.minimum((np.amax(roi_nz[1]) + max_w//2), w)
            ))

        else:
            roi_window.append((random.randint(0,64),random.randint(-64,0)))
            roi_window.append((random.randint(0,64),random.randint(-64,0)))
        direction = random.choice(['t','d','l','r','no_erase'])
        # print(direction)
        if direction == 't':
            image[:roi_window[0][0],:] = 0
        elif direction == 'd':
            image[roi_window[0][1]:,:] = 0
        elif direction == 'l':
            image[:,:roi_window[1][0]] = 0
        elif direction == 'r':
            image[:,roi_window[1][1]:] = 0

        new_sample = {'image': image, 'mask': mask}

        return new_sample





class RandomFlip2D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array
        # mask: numpy array
        image = sample['image']
        mask = sample['mask']

        image = Image.fromarray(image)
        mask = Image.fromarray(np.uint8(mask))

        if 'h' in self.mode and 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        elif 'h' in self.mode:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        elif 'v' in self.mode:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        new_sample = {'image': image, 'mask': mask}

        return new_sample


class RandomRotate2D(object):
    """
    Data augmentation method.
    Rotating the image with random degree.
    Args:
    - degree: the rotate degree from (-degree , degree)
    Returns:
    - rotated image and mask
    """

    def __init__(self, degree=[-15,-10,-5,0,5,10,15]):
        self.degree = degree

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        image = Image.fromarray(image)
        mask = Image.fromarray(np.uint8(mask))

        rotate_degree = random.choice(self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        return {'image': image, 'mask': mask}