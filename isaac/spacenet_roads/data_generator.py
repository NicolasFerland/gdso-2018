"""
A Keras data generator, suitable for use with keras.model.fit_generator()


This generator assumes the following folder structure of unzipped spacenet data:

<datadir>/RGB-PanSharpen/*.tif
         /geojson/spacenetroads/*.geojson
"""

import os
from glob import glob
import json
import rasterio
from rasterio.features import rasterize
import numpy as np
from skimage import morphology
from keras.utils import to_categorical


class SpacenetGenerator(object):
    """
    Data generator for the Spacenet satellite image data set.
    Produces one-hot encodings of binary not-road / road.
    """

    def __init__(self, datadir, tile_size=(512, 512)):

        self.datadir = datadir
        self.tile_size = np.array(tile_size)
        imgpaths = glob(os.path.join(self.datadir, 'RGB-PanSharpen', '*.tif'))
        self.images = [os.path.basename(imgpath) for imgpath in imgpaths]

    def _random_image(self):
        return np.random.choice(self.images)

    def _random_tile(self, image_name):
        """
        Choose a random tile fully contained within the given image / label
        """
        with rasterio.open(os.path.join(self.datadir, 'RGB-PanSharpen', image_name)) as img_src:
            # load the whole image and the road mask
            img_data = np.rollaxis(img_src.read(), 0, 3) / 2048.
            road_mask = self._get_road_mask(image_name, img_src)
            # cut out a tile
            x_start = np.random.randint(0, img_src.shape[0] - self.tile_size[0])
            y_start = np.random.randint(0, img_src.shape[1] - self.tile_size[1])
            img = img_data[x_start: x_start+self.tile_size[0], y_start: y_start+self.tile_size[1]]
            label = road_mask[x_start: x_start+self.tile_size[0], y_start: y_start+self.tile_size[1]]
            # turn the labels into a one-hot encoding, and return both the image
            #  and the label with an extra dimension for the batch
            label_one_hot = to_categorical(label.ravel(), 2)
            return img[np.newaxis, :], label_one_hot[np.newaxis, :]

    def _get_road_mask(self, image_name, tif, kernel=morphology.disk(10)):
        """
        Given a tif filename and the open tif file, create the mask of roads within the tif
        """
        json_basename = image_name.replace('RGB-PanSharpen', 'spacenetroads').replace('.tif', '.geojson')
        json_full_path = os.path.join(self.datadir, 'geojson', 'spacenetroads', json_basename)
        json_data = json.load(open(json_full_path, 'r'))
        road_raster = rasterize(
            [(feature['geometry'], 1) for feature in json_data['features']],
            out_shape=tif.shape, transform=tif.transform, all_touched=True
        )
        dilated_road_raster = morphology.binary_dilation(road_raster, selem=kernel).astype(int)
        return dilated_road_raster

    def random_generator(self, batch_size=1):
        while True:
            img_batch = []
            lab_batch = []
            for _ in range(batch_size):
                while True:
                    image_name = self._random_image()
                    try:
                        img, lab = self._random_tile(image_name)
                    except json.JSONDecodeError:
                        print('Problem with json file:', image_name)
                        continue
                    break
                img_batch.append(img)
                lab_batch.append(lab)
            yield np.concatenate(img_batch), np.concatenate(lab_batch)
