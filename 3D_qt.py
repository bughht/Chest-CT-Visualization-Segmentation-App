'''
Author: Harryhht
Date: 2022-02-13 20:27:12
LastEditors: Harryhht
LastEditTime: 2022-02-24 20:37:03
Description: 
'''
# -*- coding: utf-8 -*-
"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""

# Add path to library (just for examples; you do not need this)


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from skimage import measure, morphology
import scipy
import nibabel as nib
import copy
import time
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')


# 'CT/CThead/CThead.103').transpose(2, 1, 0)
t = time.time()
img = nib.load('CT/3d_images.zip/3d_images/IMG_0059.nii.gz')
print(time.time()-t)
t = time.time()
# img = nib.load('CT\\3d_images.zip\\3d_images\MASK_0031.nii.gz')
data = img.get_fdata()  # .transpose(2, 0, 1)
print(time.time()-t)
t = time.time()
header = img.header
zoom = np.array(header.get_zooms())
print(header.get_zooms())
# print(img.get_data('pixdim'))


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = scan

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(
        image, real_resize_factor, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1-binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


print(data.shape)
data, spacing = resample(
    data, zoom, [1, 1, 1])
data = data.transpose(2, 0, 1)
data = data.transpose(2, 0, 1)
print(time.time()-t)
t = time.time()
segmented_lungs = segment_lung_mask(data, False).astype('uint8')
segmented_lungs_fill = segment_lung_mask(data, True).astype('uint8')
lung_ex = segmented_lungs_fill-segmented_lungs

print(time.time()-t)
t = time.time()
# print(np.sum(segmented_lungs))
# data = scipy.ndimage.interpolation.zoom(
# data, zoom=np.min(zoom)/zoom, mode='nearest')
data_bone = copy.copy(data)
data_bone[data_bone < 300] = 0
data_bone[data_bone > 400] = 0
data_bone = data_bone.astype('uint8')

data_lung = copy.copy(data)
data_lung[data_lung < 80] = 0
data_lung[data_lung > 100] = 0
# data_lung[data_lung > 45] = 0
data_lung = data_lung.astype('uint8')


# pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
# print("Shape before resampling\t", first_patient_pixels.shape)
# print("Shape after resampling\t", pix_resampled.shape)

# d1 = np.empty(data.shape+(4,), dtype=np.ubyte)
# d1[:, :, :, 0] = 255
# d1[:, :, :, 1] = 255
# d1[:, :, :, 2] = 255
# d1[:, :, :, 3] += data_bone

# d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
# d2[:, :, :, 0] = 255
# d2[:, :, :, 1] = 0
# d2[:, :, :, 2] = 0
# d2[:, :, :, 3] += data_lung

# d3 = np.empty(data.shape+(4,), dtype=np.ubyte)
# d3[:, :, :, 0] = 255
# d3[:, :, :, 1] = 255
# d3[:, :, :, 2] = 0  # 255
# d3[:, :, :, 3] += segmented_lungs*2

d4 = np.empty(data.shape+(4,), dtype=np.ubyte)
d4[:, :, :, 0] = 255
d4[:, :, :, 1] = 0
d4[:, :, :, 2] = 255
d4[:, :, :, 3] += lung_ex*30
# d2[:, 0, 0] = [255, 0, 0, 100]
# d2[0, :, 0] = [0, 255, 0, 100]
# d2[0, 0, :] = [0, 0, 255, 100]

# print(d4.shape)

# v1 = gl.GLVolumeItem(d1, sliceDensity=1, smooth=True)
# v1.translate(-d1.shape[0]/2, - d1.shape[1]/2, -d1.shape[2]/2)
# v2 = gl.GLVolumeItem(d2, sliceDensity=2, smooth=True,)
# v2.translate(-d2.shape[0]/2, - d2.shape[1]/2, -d2.shape[2]/2)
# v3 = gl.GLVolumeItem(d3, sliceDensity=1, smooth=True)
# v3.translate(-d3.shape[0]/2, - d3.shape[1]/2, -d3.shape[2]/2)
v4 = gl.GLVolumeItem(d4, sliceDensity=1, smooth=True)
v4.translate(-d4.shape[0]/2, - d4.shape[1]/2, -d4.shape[2]/2)
# w.addItem(v4)
# w.addItem(v2)

verts, faces, _, _ = measure.marching_cubes(data, 400)

m1 = gl.GLMeshItem(vertexes=verts, faces=faces,
                   color=(0.5, 0.5, 1, 1), smooth=True,)
m1.translate(-data.shape[0]/2, - data.shape[1]/2, 0)
w.addItem(m1)

ax = gl.GLAxisItem()
w.addItem(ax)


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
