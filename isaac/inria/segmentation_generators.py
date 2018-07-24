"""
A Keras data generator, suitable for use with keras.model.fit_generator()


This generator assumes the default folder structure of unzipped Inria data:

Inria/train/images/*.tif
           /gt/*.tif
     /val/images/*.tif
         /gt/*.tif
     /test/images/*.tif
"""

import os
from glob import glob
import rasterio
import numpy as np
from keras.utils import to_categorical


class InriaGenerator(object):
    """
    Data generator for the Inria Aerial image data set
    """

    def __init__(self, datadir, tile_size=(512, 512)):

        self.datadir = datadir
        self.tile_size = np.array(tile_size)
        self.images = [os.path.basename(imgpath) for imgpath in glob(os.path.join(self.datadir, 'images', '*.tif'))]

    def _random_image(self):
        return np.random.choice(self.images)

    def _random_tile(self, image_name):
        """
        Choose a random tile fully contained within the given image / label
        """
        with rasterio.open(os.path.join(self.datadir, 'images', image_name)) as img_src:
            img_shape = img_src.shape
            r_start = np.random.randint(0, img_shape[1] - self.tile_size[1])
            c_start = np.random.randint(0, img_shape[0] - self.tile_size[0])
            window = ((r_start, r_start+self.tile_size[1]), (c_start, c_start+self.tile_size[0]))
            img = np.rollaxis(img_src.read((1,2,3), window=window) / 255, 0, 3)
            with rasterio.open(os.path.join(self.datadir, 'gt', image_name)) as lab_src:
                lab = lab_src.read((1, ), window=window)[0] // 255
            lab_one_hot = to_categorical(lab.ravel(), 2)
            return img[None], lab_one_hot[None]

    def _fully_tile(self, image_name):
        with rasterio.open(os.path.join(self.datadir, 'images', image_name)) as img_src:
            with rasterio.open(os.path.join(self.datadir, 'gt', image_name)) as lab_src:
                img_shape = img_src.shape
                for r_start in range(0, img_shape[1] - self.tile_size[1], self.tile_size[1]):
                    for c_start in range(0, img_shape[0] - self.tile_size[0], self.tile_size[0]):
                        window = ((r_start, r_start+self.tile_size[1]), (c_start, c_start+self.tile_size[0]))
                        img = np.rollaxis(img_src.read((1,2,3), window=window) / 255, 0, 3)
                        lab = lab_src.read((1, ), window=window)[0] // 255
                        lab_one_hot = to_categorical(lab.ravel(), 2)
                        yield img[None], lab_one_hot[None]

    def random_generator(self, batch_size=5):
        while True:
            img_batch = []
            lab_batch = []
            for _ in range(batch_size):
                img, lab = self._random_tile(self._random_image())
                img_batch.append(img)
                lab_batch.append(lab)
            yield np.concatenate(img_batch), np.concatenate(lab_batch)
