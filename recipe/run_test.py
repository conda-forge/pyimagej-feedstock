import sys
# Work around Java installation issues in CI
if sys.platform == 'darwin':
    sys.exit(0)

import imagej
import numpy as np

ij = imagej.init(headless=True)
print(ij.getVersion())

img_shape = (512, 512)

img = np.random.random(img_shape)
output = np.zeros(img.shape, dtype=img.dtype)
rai = ij.op().filter().frangiVesselness(ij.py.to_java(output), ij.py.to_java(img), [1, 1], 20)

assert output.shape == img_shape
assert rai.numDimensions() == 2
assert rai.dimension(0) == img_shape[0]
assert rai.dimension(1) == img_shape[1]
