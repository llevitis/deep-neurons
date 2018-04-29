import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from skimage.io import imread
from skimage import filters as skfilt
from PIL import Image

f = h5py.File(os.getcwd() + '/deep-neurons.hdf5', 'r')

def show_thresholds():
    images = f['images']
    filenames = f['filenames']
    for i in range(21, 40):
        image = images[i]
        thresholds = {
            "thresh_yen": skfilt.threshold_yen(image),
            "thresh_otsu": skfilt.threshold_otsu(image),
            "thresh_li": skfilt.threshold_li(image),
            "thresh_iso": skfilt.threshold_isodata(image)
        }
        name = filenames[i]
        fig = plt.figure(i)
        arr = np.asarray(image)
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(arr, cmap='gray')
        ax.set_title(name)
        for i_plt, thresh_type in enumerate(thresholds.keys()):
            thresh_val = thresholds[thresh_type]
            ax1 = fig.add_subplot(3, 2, i_plt + 2)
            ax1.imshow(image > thresh_val, interpolation='nearest', cmap='gray')
            ax1.set_title(thresh_type + ": {}".format(np.round(thresholds[thresh_type])))
        plt.show()


def main():
    show_thresholds()

if __name__ == "__main__":
    main()




#
# fig = plt.figure()
# for i_plt, thresh_type in enumerate(thresholds.keys()):
#     thresh_val = thresholds[thresh_type]
#     ax1 = fig.add_subplot(2, 2, i_plt + 1)
#     ax1.imshow(img_3 > thresh_val, interpolation='nearest', cmap='gray')
#     ax1.set_title(thresh_type + ": {}".format(np.round(thresholds[thresh_type])))
#
# plt.show()