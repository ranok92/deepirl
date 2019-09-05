"""
Transforms the map.png files from EWAP scenarios using their H.txt files.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.image import imread
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    'map_file',
    help='path to map.png image file containig obstacles of scenario',
    type=Path
)
parser.add_argument(
    'hmat_file',
    help='Camera homography matrix "h.txt" of scenario.',
    type=Path
)

parser.add_argument(
    '--scale',
    help='Scale of produced image. values between 10-50 are good picks.',
    type=int
)

args = parser.parse_args()

if __name__ == '__main__':
    orig_map = imread(str(args.map_file.resolve()))
    hmat = np.loadtxt(str(args.hmat_file.resolve()))

    # all pixel positions
    pixel_pos = np.indices(orig_map.shape).T.reshape(-1, 2).T
    h_pixel_pos = np.concatenate((pixel_pos, np.ones((1, pixel_pos.shape[1]))))

    transformed_map = np.matmul(hmat, h_pixel_pos)
    transformed_map /= transformed_map[-1:]
    transformed_map = transformed_map[:2:]

    # try to coerce transformed map into discrete image
    # arbitrary 1m/10 resolution below
    shifted_map = transformed_map.copy() * args.scale

    shifted_map[0:] -= shifted_map[0, 0]
    shifted_map[1:] -= shifted_map[1, 0]

    shifted_map = np.round(shifted_map).astype(np.int64)

    # construct new image
    max_x = np.max(shifted_map[0]).item() + 1
    max_y = np.max(shifted_map[1]).item() + 1

    new_image = np.zeros((max_x, max_y))

    # populate new image
    for i in range(shifted_map.shape[1]):
        old_coords = pixel_pos[:, i]
        rw_coords = shifted_map[:, i]
        new_image[tuple(rw_coords)] = orig_map[tuple(old_coords)]

    # save image
    im2save = Image.fromarray(new_image.astype('uint8')*255)
    im2save = im2save.convert('1')
    im2save.save(args.map_file.parents[0]/'hmap.png')
