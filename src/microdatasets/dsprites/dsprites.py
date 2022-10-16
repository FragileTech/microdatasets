import matplotlib.pyplot as plt
import numpy as np

from microdatasets import ASSETS_PATH


DSPRITES_PATH = ASSETS_PATH / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"


def resize_frame(frame: np.ndarray, width: int, height: int, mode: str = "L") -> np.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    from PIL import Image

    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return np.array(frame)


class DSprites:

    DEFAULT_DATA_FILE = DSPRITES_PATH

    def __init__(self, file=DEFAULT_DATA_FILE):
        imgs, latents_values, latents_classes, metadata = self.load_dataset(file)
        self.imgs = imgs
        self.latents_values = latents_values
        self.latents_classes = latents_classes
        self.metadata = metadata
        self.latents_sizes = metadata["latents_sizes"]
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1])),
        )

    # Helper function to show images
    @staticmethod
    def show_images_grid(imgs_, num_images=25):
        ncols = int(np.ceil(num_images**0.5))
        nrows = int(np.ceil(num_images / ncols))
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(imgs_[ax_i], cmap="Greys_r", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

    @staticmethod
    def show_density(imgs):
        _, ax = plt.subplots()
        ax.imshow(imgs.mean(axis=0), interpolation="nearest", cmap="Greys_r")
        ax.grid("off")
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def resize_frame(frame: np.ndarray, width: int, height: int, mode: str = "L") -> np.ndarray:
        """
        Use PIL to resize an RGB frame to an specified height and width.

        Args:
            frame: Target numpy array representing the image that will be resized.
            width: Width of the resized image.
            height: Height of the resized image.
            mode: Passed to Image.convert.

        Returns:
            The resized frame that matches the provided width and height.

        """

        return resize_frame(frame, width, height, mode)

    @classmethod
    def decode_keys(cls, dict_):
        d = {}
        for k, v in dict_.items():
            k = k.decode() if isinstance(k, bytes) else k
            if not isinstance(v, dict):
                d[k] = v.decode() if isinstance(v, bytes) else v
            else:
                d[k] = cls.decode_keys(v)
        return d

    def load_dataset(self, file):
        dataset_zip = np.load(file, allow_pickle=True, encoding="bytes")
        imgs = dataset_zip["imgs"]
        latents_values = dataset_zip["latents_values"]
        latents_classes = dataset_zip["latents_classes"]
        metadata = self.decode_keys(dataset_zip["metadata"][()])
        return imgs, latents_values, latents_classes, metadata

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size, n_shapes=None, only_big_scales=True):
        samples = np.zeros((size, self.latents_sizes.size), dtype=int)
        for lat_i, lat_size in enumerate(self.latents_sizes):
            if lat_i == 1 and n_shapes is not None:  # Limit number of shapes
                samples[:, lat_i] = np.random.randint(n_shapes, size=size)
            elif lat_i == 2 and only_big_scales:
                samples[:, lat_i] = np.random.randint(4, 6, size=size)
            else:
                samples[:, lat_i] = np.random.randint(lat_size, size=size)
        x = np.unique(samples, axis=0)
        return x[np.random.permutation(np.arange(x.shape[0]))]

    def sample_examples(self, n_examples, n_shapes=None, width=40, height=40):
        # Sample latents randomly. When image size < 64x64
        # only the two biggest scale values will be chosen
        big_scale = width < 64 or height < 64
        latents_sampled = self.sample_latent(
            size=int(n_examples * 1.5),
            n_shapes=n_shapes,
            only_big_scales=big_scale,
        )[:n_examples]
        # Select images
        imgs_sampled = self.latents_to_img(latents_sampled)
        imgs_res = np.array(
            [self.resize_frame(im, width, height, mode="L") for im in imgs_sampled],
        )
        return imgs_res, latents_sampled

    def latents_to_img(self, latents):
        indices_sampled = self.latent_to_index(latents)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

    def sample_dataset(
        self,
        n_examples,
        shapes_per_image=1,
        allow_overlap=True,
        n_shapes=None,
        width=40,
        height=40,
    ):
        if shapes_per_image == 1:
            return self.sample_examples(
                n_examples=n_examples,
                n_shapes=n_shapes,
                width=width,
                height=height,
            )
        data = [
            self.sample_examples(
                n_examples=n_examples,
                n_shapes=n_shapes,
                width=width,
                height=height,
            )
            for _ in range(shapes_per_image)
        ]
        examples, latents = list(zip(*data))
        examples = np.array(examples).sum(0)
        if allow_overlap:
            return examples, latents
        ix = examples.max((1, 2)) == 1
        latents = np.ascontiguousarray(np.array(latents)[:, ix].swapaxes(0, 2).swapaxes(0, 1))
        return examples[ix], latents

    @staticmethod
    def swap_pixels(imgs: np.ndarray, prob: float) -> np.ndarray:
        """
        Swap pixel values for an example with target probability.

        Args:
            imgs (np.ndarray): target image for swapping its pixel values.
            prob (float): probability of swapping a each pixel value of the image

        Returns:
            np.ndarray: Images with its pixels swapped.
        """
        swaps = np.random.random(imgs.shape) < prob
        inv_img = np.logical_not(imgs.astype(np.bool_)).astype(np.uint8)
        return np.where(swaps, inv_img, imgs.astype(np.uint8))
