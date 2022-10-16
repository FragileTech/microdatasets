import numpy
import pytest

from microdatasets.dsprites.dsprites import DSprites


@pytest.fixture(scope="module")
def dsprites() -> DSprites:
    return DSprites()


class TestDSprites:
    def test_init(self, dsprites):
        # Test default initialization
        pass

    def test_load_dataset(self, dsprites):
        imgs, latents_values, latents_classes, metadata = dsprites.load_dataset(
            dsprites.DEFAULT_DATA_FILE
        )
        assert isinstance(metadata, dict)

    def test_sample_dataset(self, dsprites):
        imgs, lats = dsprites.sample_dataset(
            1000, shapes_per_image=1, allow_overlap=True, n_shapes=3, width=64, height=64
        )
        assert isinstance(imgs, numpy.ndarray)
        assert isinstance(lats, numpy.ndarray)
        assert imgs.shape == (1000, 64, 64)
