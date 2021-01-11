import numpy as np
import torch
from PIL import Image,ImageOps
import random
from skimage import exposure
from skimage.util import random_noise
from skimage.transform import warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose

import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class RandomEraseHalf(object):
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

        c,h,w = image.shape
        roi_window = []
        
        if np.sum(mask) !=0:
            roi_nz = np.nonzero(mask)
            roi_window.append((
                np.maximum((np.amin(roi_nz[1]) - max_h//2), 0),
                np.minimum((np.amax(roi_nz[1]) + max_h//2), h)
            ))

            roi_window.append((
                np.maximum((np.amin(roi_nz[2]) - max_w//2), 0),
                np.minimum((np.amax(roi_nz[2]) + max_w//2), w)
            ))

        else:
            roi_window.append((random.randint(0,64),random.randint(-64,0)))
            roi_window.append((random.randint(0,64),random.randint(-64,0)))
        direction = random.choice(['t','d','l','r','no_erase'])
        # print(direction)
        if direction == 't':
            image[:,:roi_window[0][0],:] = 0
        elif direction == 'd':
            image[:,roi_window[0][1]:,:] = 0
        elif direction == 'l':
            image[:,:,:roi_window[1][0]] = 0
        elif direction == 'r':
            image[:,:,roi_window[1][1]:] = 0

        new_sample = {'image': image, 'mask': mask}

        return new_sample


class RandomTranslationRotationZoomHalf(object):
    '''
    Data augmentation method.
    Including random translation, rotation and zoom, which keep the shape of input.
    Args:
    - mode: string, consisting of 't','r' or 'z'. Optional methods and 'trz'is default.
            't'-> translation,
            'r'-> rotation,
            'z'-> zoom.
    '''
    def __init__(self, mode='trz',num_class=2):
        self.mode = mode
        self.num_class = num_class

    def __call__(self, sample):
        # image: numpy array
        # mask: numpy array
        image = sample['image']
        mask = sample['mask']
        # get transform coordinate
        img_size = image.shape
        coords0, coords1, coords2 = np.mgrid[:img_size[0], :img_size[1], :
                                             img_size[2]]
        coords = np.array([
            coords0 - img_size[0] / 2, coords1 - img_size[1] / 2,
            coords2 - img_size[2] / 2
        ])
        tform_coords = np.append(coords.reshape(3, -1),
                                 np.ones((1, np.prod(img_size))),
                                 axis=0)
        # transform configuration
        # translation
        if 't' in self.mode:
            translation = [
                0, np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ]
        else:
            translation = [0, 0, 0]

        # rotation
        if 'r' in self.mode:
            rotation = euler2mat(
                np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
        else:
            rotation = euler2mat(0, 0, 0, 'sxyz')

        # zoom
        if 'z' in self.mode:
            zoom = [
                1, np.random.uniform(0.9, 1.1),
                np.random.uniform(0.9, 1.1)
            ]
        else:
            zoom = [1, 1, 1]

        # compose
        warp_mat = compose(translation, rotation, zoom)

        # transform
        w = np.dot(warp_mat, tform_coords)
        w[0] = w[0] + img_size[0] / 2
        w[1] = w[1] + img_size[1] / 2
        w[2] = w[2] + img_size[2] / 2
        warp_coords = w[0:3].reshape(3, img_size[0], img_size[1], img_size[2])

        image = warp(image, warp_coords)
        new_mask = np.zeros(mask.shape, dtype=np.float32)
        for z in range(1,self.num_class):
            temp = warp((mask == z).astype(np.float32),warp_coords)
            new_mask[temp >= 0.5] = z
        mask = new_mask   
        new_sample = {'image': image, 'mask': mask}

        return new_sample


class RandomFlipHalf(object):
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
        # image: numpy array, (D,H,W)
        # mask: integer, 0,1,..
        image = sample['image']
        mask = sample['mask']

        if 'h' in self.mode and 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, ...]
                mask = mask[:, ::-1, ...]
            else:
                image = image[..., ::-1]
                mask = mask[..., ::-1]

        elif 'h' in self.mode:
            image = image[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        elif 'v' in self.mode:
            image = image[..., ::-1]
            mask = mask[..., ::-1]
        # avoid the discontinuity of array memory
        image = image.copy()
        mask = mask.copy()
        new_sample = {'image': image, 'mask': mask}

        return new_sample


class RandomAdjustHalf(object):
    """
    Data augmentation method.
    Adjust the brightness of the image with random gamma.
    Args:
    - scale: the gamma from the scale
    Returns:
    - adjusted image
    """

    def __init__(self, scale=(0.2,1.8)):
        assert isinstance(scale,tuple)
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        gamma = random.uniform(self.scale[0],self.scale[1])
        seq_len = image.shape[0]
        for i in range(seq_len):
            image[i] = exposure.adjust_gamma(image[i], gamma) 
        sample['image'] = image
        return sample


class RandomNoiseHalf(object):
    """
    Data augmentation method.
    Add random salt-and-pepper noise to the image with a probability.
    Returns:
    - adjusted image
    """
    def __call__(self, sample):
        image = sample['image']
        prob = random.uniform(0,1)
        seq_len = image.shape[0]
        if prob > 0.9:
            for i in range(seq_len):
                image[i] = random_noise(image[i],mode='s&p') 
        sample['image'] = image
        return sample


class RandomDistortHalf(object):
    """
    Data augmentation method.
    Add random salt-and-pepper noise to the image with a probability.
    Returns:
    - adjusted image
    """
    def __init__(self,random_state=None,alpha=200,sigma=20,grid_scale=4):
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.grid_scale = grid_scale

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        seq_len = image.shape[0]
        if self.random_state is None:
            random_state = np.random.RandomState(None)

        im_merge = np.concatenate((image, mask), axis=0)
        im_merge = np.transpose(im_merge,(1,2,0)) #(H,W,2*C)
        shape = im_merge.shape
        shape_size = shape[:2]

        self.alpha //= self.grid_scale  
        self.sigma //= self.grid_scale  # more similar end result when scaling grid used.
        grid_shape = (shape_size[0]//self.grid_scale, shape_size[1]//self.grid_scale)

        blur_size = int(4 * self.sigma) | 1
        rand_x = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=self.sigma) * self.alpha
        rand_y = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=self.sigma) * self.alpha
        if self.grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)

        distorted_img = cv2.remap(im_merge, grid_x, grid_y, borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
        distorted_img = np.transpose(distorted_img,(2,0,1))
        sample['image'] = distorted_img[:seq_len]
        sample['mask']  = distorted_img[seq_len:]

        return sample
