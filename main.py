'''
Author: Harryhht
Date: 2022-02-14 22:50:37
LastEditors: Harryhht
LastEditTime: 2022-03-02 12:48:15
Description: CT Segmentation
'''

import nibabel as nib
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import math
import cv2 as cv
import copy
import scipy
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication
from Layout import Ui_MainWindow
import numpy as np
from skimage import measure
from scipy.ndimage.filters import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


warnings.filterwarnings('ignore')
print('Tensorflow version : {}'.format(tf.__version__))


class mainwin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None) -> None:
        super(mainwin, self).__init__(parent)
        self.setupUi(self)
        self.label_filename.setWordWrap(True)

        self.Var_init()
        self.Widget_UNET_init()
        self.Widget_2D_init()
        self.Widget_3D_init()
        self.Slot_init()
        self.SliderBar_init()
        self.Segmentation_model_init()

    def Var_init(self):
        self.file_dir = None
        self.Slice_TYPE_ = {0: '矢状面', 1: '冠状面', 2: '横断面'}
        self.Slice_data = None
        self.Slice_TYPE = 0
        self.ISO_select = -300
        self.COLOR = QColor(255, 255, 255, 10)
        self.PLOT_TYPE = "骨骼"

    def Segmentation_model_init(self):
        class FixedDropout(tf.keras.layers.Dropout):
            def _get_noise_shape(self, inputs):
                if self.noise_shape is None:
                    return self.noise_shape

                symbolic_shape = tf.keras.backend.shape(inputs)
                noise_shape = [symbolic_shape[axis] if shape is None else shape
                               for axis, shape in enumerate(self.noise_shape)]
                return tuple(noise_shape)

        def DiceCoef(y_trues, y_preds, smooth=1e-5, axis=None):
            intersection = tf.reduce_sum(y_trues * y_preds, axis=axis)
            union = tf.reduce_sum(y_trues, axis=axis) + \
                tf.reduce_sum(y_preds, axis=axis)
            return tf.reduce_mean((2*intersection+smooth) / (union + smooth))

        def DiceLoss(y_trues, y_preds):
            return 1.0 - DiceCoef(y_trues, y_preds)

        get_custom_objects().update(
            {'swish': tf.keras.layers.Activation(tf.nn.swish)})
        get_custom_objects().update({'FixedDropout': FixedDropout})
        get_custom_objects().update({'DiceCoef': DiceCoef})
        get_custom_objects().update({'DiceLoss': DiceLoss})
        print('Load segmentation model...')
        self.model = tf.keras.models.load_model(
            'ct/osic_segmentation_model.h5')
        print(self.model.summary())

    def Widget_UNET_init(self):
        self.figure = plt.figure()
        self.wmatplot = FigureCanvas(self.figure,)
        self.Layout_UNET.addWidget(self.wmatplot)

    def Widget_3D_init(self):
        self.w3d = gl.GLViewWidget()
        self.Layout_3D.addWidget(self.w3d)
        self.w3d.opts['distance'] = 500
        self.w3d.show()

    def Widget_2D_init(self):
        self.win = pg.GraphicsLayoutWidget()
        self.Layout_2D.addWidget(self.win)
        self.p1 = self.win.addPlot(title="")
        # Item for displaying image data
        self.img = pg.ImageItem(axisOrder='col-major')
        # self.img.setScaledMode()
        self.p1.addItem(self.img)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [100, 100])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.p1.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        self.iso = pg.IsocurveItem(level=0.6, pen='g')
        self.iso.setParentItem(self.img)
        self.iso.setZValue(15)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.setFixedWidth(100)
        self.hist.setHistogramRange(-1000, 1000)
        self.win.addItem(self.hist)

        # Draggable line for setting isocurve level
        self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.hist.vb.addItem(self.isoLine)
        self.hist.vb.setMouseEnabled(y=False)
        self.isoLine.setValue(self.ISO_select)
        self.isoLine.setZValue(1000)  # bring iso line above contrast controls

        # Another plot area for displaying ROI data
        self.win.nextRow()
        self.p2 = self.win.addPlot(colspan=2)
        self.p2.setMaximumHeight(130)
        self.win.show()

    def SliderBar_init(self):
        self.SHAPE_Update()
        self.Slider_1.setMinimum(0)
        self.Slider_1.setMaximum(self.shape[0]-1)
        self.Slider_2.setMinimum(0)
        self.Slider_2.setMaximum(self.shape[1]-1)
        self.Slider_3.setMinimum(0)
        self.Slider_3.setMaximum(self.shape[2]-1)

    def Slot_init(self):
        self.Button_LoadFIle.clicked.connect(self.Open_file)
        self.Button_FILTER.clicked.connect(self.FILTER_Update)
        self.Button_RESAMPLE.clicked.connect(self.RESAMPLE_Update)
        self.Slider_1.valueChanged.connect(self.Slider1_Update)
        self.Slider_2.valueChanged.connect(self.Slider2_Update)
        self.Slider_3.valueChanged.connect(self.Slider3_Update)
        self.Button_COLOR.clicked.connect(self.Select_COLOR)
        self.comboBox.currentIndexChanged.connect(self.Combox_Update)
        self.Button_Voxel_PLOT.clicked.connect(self.VOXEL_Gen_Plot)
        self.Button_UNET.clicked.connect(self.Predict_Lung)

        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.isoLine.sigDragged.connect(self.updateIsocurve)
        self.img.hoverEvent = self.imageHoverEvent

    def updateIsocurve(self):
        self.ISO_select = self.isoLine.value()
        self.iso.setLevel(self.ISO_select)
        self.DATAINFO_Update()

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.Slice_data.shape[0] - 1))
        j = int(np.clip(j, 0, self.Slice_data.shape[1] - 1))
        val = self.Slice_data[j, i]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.p1.setTitle("pixel: (%d, %d)  value: %g" %
                         (i, j, val))

    # Callbacks for handling user interaction
    def updatePlot(self):
        selected = self.roi.getArrayRegion(self.Slice_data, self.img)
        hist, bins = np.histogram(selected, bins=100)
        self.p2.plot(bins[1:], hist, clear=True)

    def RESAMPLE_Update(self):
        self.data, self.zoom = self.resample(self.data, self.zoom)
        self.SHAPE_Update()
        self.SliderBar_init()
        self.DATAINFO_Update()
        self.Slider3_Update()

    def FILTER_Update(self):
        # self.data = uniform_filter(self.data)
        self.data = gaussian_filter(self.data, sigma=1)
        self.SHAPE_Update()
        self.SliderBar_init()
        self.DATAINFO_Update()
        self.Slider3_Update()

    def SHAPE_Update(self):
        try:
            self.shape = self.data.shape
        except:
            self.shape = (0, 0, 0)

    def DATAINFO_Update(self):
        self.DATAINFO_TEXT = "DATA INFO:\n"
        self.DATAINFO_SHAPE = "Shape:\t"
        try:
            self.DATAINFO_SHAPE += ("[" + str(self.shape[0]) +
                                    "," + str(self.shape[1]) +
                                    "," + str(self.shape[2]) + "]\n")
        except:
            self.DATAINFO_SHAPE += "\n"
        self.DATAINFO_TEXT += self.DATAINFO_SHAPE

        self.DATAINFO_ZOOM = "Zoom:\t"
        try:
            self.DATAINFO_ZOOM += ("[" + str(round(self.zoom[0], 2)) +
                                   "," + str(round(self.zoom[1], 2)) +
                                   "," + str(round(self.zoom[2], 2)) + "]\n")
        except:
            self.DATAINFO_ZOOM += "\n"
        self.DATAINFO_TEXT += self.DATAINFO_ZOOM

        self.DATAINFO_ISO = "ISO:\t"
        try:
            self.DATAINFO_ISO += str(round(self.ISO_select))
        except:
            self.DATAINFO_ISO += "\n"
        self.DATAINFO_TEXT += self.DATAINFO_ISO

        self.label_DATAINFO.setText(self.DATAINFO_TEXT)

    def Slider1_Update(self):
        self.SLICE_LAYER_0 = self.Slider_1.value()
        self.Label_Slice1.setText("冠状面["+str(self.SLICE_LAYER_0)+"]")
        self.Slice_TYPE = 0
        self.Slice_data = self.data[self.SLICE_LAYER_0, :, :]
        self.img.setImage(self.Slice_data)
        self.updateIsocurve()
        self.iso.setData(self.Slice_data)

    def Slider2_Update(self):
        self.SLICE_LAYER_1 = self.Slider_2.value()
        self.Label_Slice2.setText("矢状面["+str(self.SLICE_LAYER_1)+"]")
        self.Slice_TYPE = 1
        self.Slice_data = self.data[:, self.SLICE_LAYER_1, :]
        self.img.setImage(self.Slice_data)
        self.updateIsocurve()
        self.iso.setData(self.Slice_data)

    def Slider3_Update(self):
        self.SLICE_LAYER_2 = self.Slider_3.value()
        self.Label_Slice3.setText("横断面["+str(self.SLICE_LAYER_2)+"]")
        self.Slice_TYPE = 2
        self.Slice_data = self.data[:, :, self.SLICE_LAYER_2].T
        self.img.setImage(self.Slice_data)
        self.updateIsocurve()
        self.iso.setData(self.Slice_data)

    def Combox_Update(self, type):
        self.PLOT_TYPE = self.comboBox.currentText()
        print(self.PLOT_TYPE)

    def Open_file(self):
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'E:\SHU\医学信号处理\Project\CODE\CT\\3d_images.zip\\3d_images', "Image files (*.nii *.nii.gz)")
        self.file_dir = fname[0]
        self.label_filename.setText(self.file_dir)
        self.file = nib.load(self.file_dir)
        self.header = self.file.header
        self.zoom = np.array(self.header.get_zooms())
        print(self.zoom)
        self.zoom = np.roll(self.zoom, -1)
        print(self.zoom)
        self.data = self.file.get_fdata()

        self.data = self.data.transpose(1, 2, 0)
        self.SHAPE_Update()
        self.SliderBar_init()
        self.DATAINFO_Update()

    def Select_COLOR(self):
        self.COLOR = QColorDialog.getColor()
        self.COLOR.setAlpha(10)
        print(self.COLOR.getRgb())

    def Load_Slices(self, BS):
        DIM = 256
        images = np.zeros(
            (BS, DIM, DIM, 3), dtype=np.uint8)
        for idx in range(BS):
            image = self.data[:, :, idx*math.floor(self.shape[2]/BS)]
            image = ((image - np.min(image)) / (np.max(image) -
                     np.min(image)) * 255).astype(np.uint8)
            image = cv.resize(
                image, (DIM, DIM), cv.INTER_AREA)
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            images[idx] = cv.flip(image, 0)
        return images/255.0

    def Predict_Lung(self):
        self.images = self.Load_Slices(20)
        self.pred_masks = self.model.predict(self.images, verbose=0)
        self.pred_masks = (self.pred_masks > 0.5).astype(np.float32)

        for idx, (image, mask) in enumerate(zip(self.images, self.pred_masks)):
            axis = self.figure.add_subplot(
                2, 10, idx+1)
            axis.imshow(image)
            axis.imshow(mask[:, :, 0], alpha=0.35)
            axis.set_xticks([])
            axis.set_yticks([])
        self.wmatplot.draw()
        # plt.subplot(15, math.ceil(self.shape[2]/10), idx+1)
        #     plt.imshow(image)
        #     plt.imshow(mask[:, :, 0], alpha=0.35)
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()

    def ISO_Surface_Gen_Plot(self):
        # try:
        #     self.w3d.removeItem(self.m1)
        # except:
        #     pass
        self.verts, self.mesh, _, _ = measure.marching_cubes(
            self.data, self.ISO_select)

        self.m1 = gl.GLMeshItem(vertexes=self.verts,
                                faces=self.mesh, color=np.array(self.COLOR.getRgb()), shader="balloon")
        self.m1.translate(-self.shape[0]/2, -self.shape[1]/2, -self.shape[2]/2)
        self.w3d.addItem(self.m1)

    def VOXEL_Gen_Plot(self):
        self.w3d.clear()
        if self.PLOT_TYPE == "骨骼":
            self.data_display = np.bitwise_and(
                self.data < 400, self.data > 300).astype('uint8')*30

        elif self.PLOT_TYPE == "肺叶":
            self.data_display = self.segment_lung_mask(
                self.data, False)*2
            self.data_display = self.data_display.astype('uint8')

        elif self.PLOT_TYPE == "支气管":
            self.data_display = (self.segment_lung_mask(
                self.data, True)-self.segment_lung_mask(self.data, False))*30
            self.data_display = self.data_display.astype('uint8')

        self.d = np.empty(self.data_display.shape+(4,), dtype=np.ubyte)
        print(self.d.shape)
        self.d[:, :, :, 0] += np.array(self.COLOR.getRgb())[0]
        self.d[:, :, :, 1] += np.array(self.COLOR.getRgb())[1]
        self.d[:, :, :, 2] += np.array(self.COLOR.getRgb())[2]
        self.d[:, :, :, 3] += self.data_display
        self.v = gl.GLVolumeItem(self.d, sliceDensity=1, smooth=True)
        self.v.translate(*(-np.array(self.d.shape)/2))
        self.w3d.addItem(self.v)

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = scan
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(
            image, real_resize_factor, mode='nearest', prefilter=False)

        return image, new_spacing

    def largest_label_volume(self, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def segment_lung_mask(self, image, fill_lung_structures=True):
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
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1-binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image.astype('uint8')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = mainwin()
    ui.show()
    sys.exit(app.exec_())
