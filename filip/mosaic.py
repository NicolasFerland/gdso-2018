import numpy as np
from numpy import random
import json
import rasterio
import os
from rasterio import merge
from rasterio import mask
from rasterio import transform
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

class Mosaic:
    def __init__(self, rootdir):
        self.imgdir = rootdir + "/RGB-PanSharpen/"
        self.roaddir = rootdir + "/roads/"
        files = []
        coords = []
        for filename in os.listdir(self.imgdir):
            if(filename.endswith(".tif")):
                current = rasterio.open(self.imgdir + filename, "r")
                files.append(filename)
                coords.append(current.transform * (0, 0))
                current.close()

        self.images = files
        self.coords = coords

        x_coords, y_coords = zip(*coords)
        self.minx, self.maxx = min(x_coords), max(x_coords)
        self.miny, self.maxy = min(y_coords), max(y_coords)
        #print(x_coords, self.minx)

        self.stepx = min([x - self.minx for x in x_coords if x > self.minx])
        self.stepy = min([y - self.miny for y in y_coords if y > self.miny])

        gridx = (np.array([x - self.minx for x in x_coords]) / self.stepx).astype(int)
        gridy = (np.array([y - self.miny for y in y_coords]) / self.stepy).astype(int)

        self.grid = list(zip(gridx, gridy))

    #give the name of the road file corresponding to tifname file
    def roadName(self, tifname):
        return "road" + tifname[30:-4] + ".tif"

    #give the number of the image at location (grid_x, grid_y) in mosaic
    def imgNumber(self, grid_x, grid_y):
        if (grid_x, grid_y) in self.grid:
            return self.grid.index((grid_x, grid_y))
        else:
            return False

    #give the name of the image at location (grid_x, grid_y) in mosaic
    def imgName(self, grid_x, grid_y):
        if (grid_x, grid_y) in self.grid:
            return self.images[self.imgNumber(grid_x, grid_y)]
        else:
            return False

    #Give the grid position of the putative image containing coordinates (x,y)
    def gridWithCoords(self, x, y):
        grid_x = int(np.floor((x - self.minx) / self.stepx))
        grid_y = int(np.floor((y - self.miny) / self.stepy)) + 1
        return [grid_x, grid_y]


    #Give the name of the image containing coordinates (x,y)
    def imgWithCoords(self, x, y):
        grid_x = int(np.floor((x - self.minx) / self.stepx))
        grid_y = int(np.floor((y - self.miny) / self.stepy)) + 1
        if self.imgNumber(grid_x, grid_y):
            return self.imgName(grid_x, grid_y)

    #Give the grid coordinates of a given image
    def gridWithImage(self, filename):
        return self.grid[self.images.index(os.path.basename(filename))]

    #Give coordinates of four corners of the square
    #Input: coordinates of the top left corner, rotation angle, size of image in pixels
    def defineSquare(self, x, y, alpha, length):
        transform0 = rasterio.open(self.imgdir + self.imgWithCoords(x, y), 'r').transform
        x_ras, y_ras = np.round((~transform0) * (x, y))
        transform1 = transform0 * A.translation(x_ras, y_ras) * A.rotation(alpha)
        corners = [transform1 * (0, 0), transform1 * (length, 0), transform1 * (length, length), transform1 * (0, length)]
        return corners

    #Give all (possibly redundant) images containing the square
    def imagesFromSquare(self, square):
        x, y = zip(*square)
        grid_x_min = int(np.floor((min(x) - self.minx) / self.stepx))
        grid_x_max = int(np.floor((max(x) - self.minx) / self.stepx))
        grid_y_min = int(np.floor((min(y) - self.miny) / self.stepy)) + 1
        grid_y_max = int(np.floor((max(y) - self.miny) / self.stepy)) + 1
        imgs = []
        for i in range(grid_x_min, grid_x_max + 1):
            for j in range(grid_y_min, grid_y_max + 1):
                if self.imgNumber(i,j):
                    imgs.append(self.imgName(i, j))
        return imgs

    #Give the image defined by square
    def newImage(self, x, y, alpha, length):
        square = self.defineSquare(x, y, alpha, length)
        imgs = self.imagesFromSquare(square)
        if len(imgs) < 1:
            return 0
        datasets = []
        road_datasets = []
        for filename in imgs:
            datasets.append(rasterio.open(self.imgdir + filename, 'r'))
            road_datasets.append(rasterio.open(self.roaddir + self.roadName(filename), 'r'))

        if len(datasets) > 1:
            mergedImages, src_transform = rasterio.merge.merge(datasets)
            mergedRoads = rasterio.merge.merge(road_datasets)[0]
        else:
            mergedImages = datasets[0].read()
            mergedRoads = road_datasets[0].read()
            src_transform = datasets[0].transform

        #define destination transform from the upper left corner of the square
        translationVector = ~src_transform * square[0]
        dst_transform = src_transform * A.translation(*translationVector) * A.rotation(alpha)

        dest = np.zeros((mergedImages.shape[0], length, length), mergedImages.dtype)
        road_dest = np.zeros((length, length), mergedRoads.dtype)

        reproject(
            mergedImages,
            dest,
            src_transform=src_transform,
            src_crs=datasets[0].crs,
            dst_transform=dst_transform,
            dst_crs=datasets[0].crs,
            resampling=Resampling.nearest)

        reproject(
            mergedRoads,
            road_dest,
            src_transform=src_transform,
            src_crs=datasets[0].crs,
            dst_transform=dst_transform,
            dst_crs=datasets[0].crs,
            resampling=Resampling.nearest)

        return [dest, road_dest, dst_transform]

    def cropImage(self, length, x = None, y = None, alpha = None):
            if x == None:
                x = self.minx + np.random.random() * (self.maxx - self.minx)
            if y == None:
                y = self.miny + np.random.random() * (self.maxy - self.miny)
            if alpha == None:
                alpha = 90. * np.random.random()

            print(x, y, alpha)
            
            return self.newImage(x, y, alpha, length)
