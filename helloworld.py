'''
Author: Harryhht
Date: 2022-02-10 23:46:43
LastEditors: Harryhht
LastEditTime: 2022-02-14 23:31:15
Description: Hello World
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib


# data = np.load('CT\scans_128x128\ID00010637202177584971671_hu_scans.npy')
# img = nib.load('CT/3d_images.zip/3d_images/IMG_0002.nii.gz')
# mask = nib.load('CT/3d_images.zip/3d_images/MASK_0002.nii.gz')
img = nib.load('MRI\\195021\Sub0001_Ses1\Sub0001_Ses1_Scan_01_ANAT1.nii.gz')

print(img.header)

epi_img_data = img.get_fdata()
# epi_mask_data = mask.get_fdata()
# print(epi_img_data.shape, epi_mask_data.shape)


def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# plot_3d(epi_img_data, 400)
# plt.hist(epi_img_data.flatten(), bins=80, color='c')
# plt.show()

plt.imshow(epi_img_data[100], cmap=plt.cm.gray)
plt.show()
