from dsprites import resize_frame
import numpy as np
from scipy import ndimage


def two_triangle_shape():
    b = np.ones((26, 26))
    new_shape = np.tril(b)[::-1] * np.tril(b) + np.triu(b)[::-1] * np.triu(b)
    new_shape[0, :] = 0
    new_shape[-1, :] = 0
    new_shape[:, 0] = 0
    new_shape[:, -1] = 0
    new_shape = new_shape.T
    return new_shape


def one_triangle():
    new_shape = np.tril(np.ones((26, 26)))
    new_shape = new_shape.T
    new_shape[0, :] = 0
    new_shape[-1, :] = 0
    new_shape[:, 0] = 0
    new_shape[:, -1] = 0
    return new_shape


class ShapeGenerator:
    SHAPES = {0: one_triangle(), 1: two_triangle_shape()}
    SCALES = np.linspace(15, 35, 6, dtype=int)
    ANGLES = np.linspace(0, 355, 40)

    @staticmethod
    def rotate_shape(shape, degs):
        ro = ndimage.rotate(shape, degs, prefilter=False, order=1)
        ro = np.where(ro > 0, 1, 0)
        ro[0, :] = 0
        ro[-1, :] = 0
        ro[:, 0] = 0
        ro[:, -1] = 0
        return ro.astype(np.float32)

    @staticmethod
    def resize_shape(shape, size):
        return resize_frame(shape, size, size).astype(np.uint8)

    @staticmethod
    def place_shape(shape, x, y):
        bg = np.zeros((64, 64))
        x_vals = np.linspace(0, bg.shape[0] - shape.shape[0], 32, dtype=int)
        y_vals = np.linspace(0, bg.shape[1] - shape.shape[1], 32, dtype=int)
        bg[x_vals[x] : x_vals[x] + shape.shape[0], y_vals[y] : y_vals[y] + shape.shape[1]] = shape
        return bg

    def latent_to_image(self, latent):
        color, shape_ix, scale, pos_x, pos_y, orientation = latent
        shape = self.SHAPES[shape_ix]
        shape = self.rotate_shape(shape, self.ANGLES[orientation])
        shape = self.resize_shape(shape, self.SCALES[scale])
        return self.place_shape(shape, pos_x, pos_y)
