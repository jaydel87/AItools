"""
From https://pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
"""

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
# load the image and convert it to a floating point data type
# image = img_as_float(io.imread(args["image"]))
image = img_as_float(io.imread("C:/Users/jayde/IONOS HiDrive/R&D/PROJETS/BFTE_SPARSE/Datasets/WSSS4LUAD/WSSS4LUAD/2.validation/2.validation/img/34.png"))
print(io.imread("C:/Users/jayde/IONOS HiDrive/R&D/PROJETS/BFTE_SPARSE/Datasets/WSSS4LUAD/WSSS4LUAD/2.validation/2.validation/img/34.png")[0, 0, :])
# loop over the number of segments
for c in (7, 8, 9, 10, 11, 12, 13, 14, 15):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments=1000, sigma=2, compactness = c, start_label=1)
	print(segments.shape)
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %f" % (c))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
# show the plots
plt.show()