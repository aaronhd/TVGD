import os
from datetime import datetime

import h5py
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate, Dropout, Multiply
from keras.layers import Input
from keras.models import Model
from keras.optimizers import *
from keras import losses
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import provider

print("Tensorflow Version: %s" % str(tf.__version__))
CURRENT_FILE, _ = os.path.basename(__file__).split('.')
FINE_TUNE = False
MULTI_GPU = True
Pre_model_used = True
FILTER_SIZES = [
    [(9, 9), (5, 5), (3, 3)]
]

NO_FILTERS = [
    [32, 16, 8],
]


INPUT_DATASET = '/media/aarons/hdd_2/corrnel_grasp_dataset/hdf5/RGB-F/dataset_200408_1444.hdf5'  #

if Pre_model_used:
    PRETRAIN_MODEL = '/media/aarons/hdd_2/background_learning_model/networks/190804_1446__background_learning/epoch_20_model.hdf5' # full data 2/3 336*336 RGBD
else:
    PRETRAIN_MODEL = None

zipdata = 'float16'
width_max = 241.0 #236.0

def load_data():
    # =====================================================================================================
    # Load the data.
    global width_max
    print('INP:', INPUT_DATASET, width_max)
    print('PRE_MODEL', PRETRAIN_MODEL)
    f = h5py.File(INPUT_DATASET, 'r')

    depth_train = np.expand_dims(np.array(f['train/depth_inpainted'], dtype=zipdata), -1)
    point_train = np.expand_dims(np.array(f['train/grasp_points_img'], dtype=zipdata), -1)
    angle_train = np.array(f['train/angle_img'], dtype=zipdata)
    cos_train = np.expand_dims(np.cos(2*angle_train), -1)
    sin_train = np.expand_dims(np.sin(2*angle_train), -1)
    grasp_width_train = np.expand_dims(np.array(f['train/grasp_width'], dtype=zipdata), -1)
    # grey_train = np.expand_dims(np.array(f['train/grey'], dtype='float16'), -1)
    rgb_train = np.array(f['train/rgb'], dtype=zipdata)
    force_train = np.expand_dims(np.array(f['train/force_img'], dtype=zipdata), -1)

    # print(depth_train.shape)
    # print(grey_train.shape)
    # exit()
    depth_test = np.expand_dims(np.array(f['test/depth_inpainted'], dtype=zipdata), -1)
    point_test = np.expand_dims(np.array(f['test/grasp_points_img'], dtype=zipdata), -1)
    angle_test = np.array(f['test/angle_img'], dtype=zipdata)
    cos_test = np.expand_dims(np.cos(2*angle_test), -1)
    sin_test = np.expand_dims(np.sin(2*angle_test), -1)
    grasp_width_test = np.expand_dims(np.array(f['test/grasp_width'], dtype=zipdata), -1)
    # grey_test = np.expand_dims(np.array(f['test/grey'], dtype='float16'), -1)
    rgb_test = np.array(f['test/rgb'], dtype=zipdata)
    force_test = np.expand_dims(np.array(f['test/force_img'], dtype=zipdata), -1)

    # for i in range(10):
    #     print(angle_train.shape)
    #     provider.show_image(np.squeeze(angle_train[i]))
    # Ground truth bounding boxes.
    # gt_bbs = np.array(f['test/bounding_boxes'], dtype='float16')
    f.close()
    # print(grasp_width_train.shape)
    # # raw_input()
    # a = angle_train[0]
    # a = np.squeeze(a)
    # print(a.shape)
    # print(np.max(a), np.min(a))
    # provider.show_image(a)
    # ====================================================================================================
    # Set up the train and test data.

    # x_train = depth_train
    # x_train = np.concatenate((grey_train, depth_train), axis=3)
    x_train2 = [depth_train, rgb_train]
    grasp_width_train = np.clip(grasp_width_train, 0, width_max)/width_max
    y_train = [point_train, cos_train, sin_train, grasp_width_train, depth_train, force_train]

    # x_test = depth_test
    # x_test = np.concatenate((grey_test, depth_test), axis=3)
    x_test2 = [depth_test, rgb_test]
    grasp_width_test = np.clip(grasp_width_test, 0, width_max)/width_max
    y_test = [point_test, cos_test, sin_test, grasp_width_test, depth_test, force_test]

    return x_train2, y_train, x_test2, y_test

    # ======================================================================================================================


def depth_loss(y_true, y_pred):
    mse_part = losses.mean_squared_error(y_true, y_pred)
    mbe_part = losses.mean_absolute_error(y_true, y_pred)
    part2 = 0.25 * mbe_part ** 2
    d = y_true - y_pred

    graXY = tf.image.sobel_edges(d)
    part3 = tf.reduce_mean(tf.square(graXY))

    return mse_part + part3


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def load_pretrain_model(model_file, out_layer_name='decoder-3'):
    model = load_model(model_file)
    if out_layer_name:
        clip_model = Model(inputs=model.input, outputs=model.get_layer(out_layer_name).output)
        return clip_model
    else:
        return model


def fusion_net(input_layer_1):
    input_layer = input_layer_1
    conv1_confi = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='fusion_conv1')(
        input_layer)
    conv2_confi = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='fusion_conv2')(
        conv1_confi)
    conv3_confi = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='fusion_conv3')(
        conv2_confi)
    conv4_confi = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='fusion_conv4')(
        conv3_confi)
    conv5_confi = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='fusion_conv5')(
        conv4_confi)
    scale1_confi = Conv2D(1, 1, activation='linear', padding='same', kernel_initializer='he_normal', name='scale_1')(
        conv5_confi)
    scale2_confi = MaxPooling2D(pool_size=(2, 2), name='scale_2')(scale1_confi)
    scale3_confi = MaxPooling2D(pool_size=(2, 2), name='scale_3')(scale2_confi)
    scale4_confi = MaxPooling2D(pool_size=(2, 2), name='scale_4')(scale3_confi)
    return scale1_confi, scale2_confi, scale3_confi, scale4_confi


def FCN_net(input_layer_1, input_layer_2):
    input_layer_1 = input_layer_1
    input_layer_2 = input_layer_2
    # input_layer = Input(shape=(480, 480, 1))
    # print(x_train.shape[1:])

    x_1 = Conv2D(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu',
                 name='1-encoder1', trainable=False)(input_layer_1)
    x_1 = Conv2D(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu',
                 name='1-encoder2', trainable=False)(x_1)
    encoded_1 = Conv2D(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu',
                       name='1-encoder3', trainable=False)(x_1)

    x_2 = Conv2D(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu',
                 name='2-encoder1', trainable=False)(input_layer_2)
    x_2 = Conv2D(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu',
                 name='2-encoder2', trainable=False)(x_2)
    encoded_2 = Conv2D(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu',
                       name='2-encoder3', trainable=False)(x_2)

    encoded = concatenate([encoded_1, encoded_2], axis=3, name='concate')

    x = Conv2DTranspose(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu',
                        name='decoder-1', trainable=False)(encoded)
    x = Conv2DTranspose(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu',
                        name='decoder-2', trainable=False)(x)
    x_FCN = Conv2DTranspose(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu',
                        name='decoder-3', trainable=True)(x)
    return x_FCN


def UG_Net(input_shape=(336, 336, 2)):
    # U-Network
    input_layer_1 = Input(shape=(336, 336, 1))  #depth
    input_layer_2 = Input(shape=(336, 336, 3))  #RGB
    # conv6_depth_1, conv7_depth_1, conv8_depth_1, conv9_depth_1 = depth_U_Net(input_layer_1)

    x_FCN = FCN_net(input_layer_1, input_layer_2)

    # input_layer = Input(shape=input_shape)
    # print(input_layer)
    # input_layer = Input(shape=x_train.shape[1:])
    # print(x_train.shape[1:])

    # depth branch
    # conv6_depth, conv7_depth, conv8_depth, conv9_depth = depth_U_Net(input_layer_1)
    scale1, scale2, scale3, scale4 = fusion_net(input_layer_1)

    # encoder part
    # RGB banch
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_layer_2)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # depth branch
    conv1_depth = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_layer_1)
    conv1_depth = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_depth')(conv1_depth)
    pool1_depth = MaxPooling2D(pool_size=(2, 2))(conv1_depth)
    conv2_depth = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_depth)
    conv2_depth = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_depth')(conv2_depth)
    pool2_depth = MaxPooling2D(pool_size=(2, 2))(conv2_depth)
    conv3_depth = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_depth)
    conv3_depth = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_depth')(conv3_depth)
    pool3_depth = MaxPooling2D(pool_size=(2, 2))(conv3_depth)
    conv4_depth = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_depth)
    conv4_depth = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_depth')(conv4_depth)
    drop4_depth = Dropout(0.5)(conv4_depth)
    pool4_depth = MaxPooling2D(pool_size=(2, 2))(drop4_depth)
    conv5_depth = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_depth)
    conv5_depth = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_depth')(conv5_depth)
    drop5_depth = Dropout(0.5)(conv5_depth)

    # decoder
    # ======6 scale======
    up6_depth = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5_depth))
    merge6_depth = concatenate([drop4_depth, up6_depth], axis=3)
    conv6_depth = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6_depth)
    conv6_depth = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6_depth')(conv6_depth)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6_depth = Multiply()([scale4, conv6_depth])
    conv6_RGBD = concatenate([conv6, conv6_depth], axis=3)
    # conv6_RGBD = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv6_RGBD)

    # ======7 scale======
    up7_depth = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6_depth))
    merge7_depth = concatenate([conv3_depth, up7_depth], axis=3)
    conv7_depth = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_depth)
    conv7_depth = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7_depth')(conv7_depth)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6_RGBD))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7_depth = Multiply()([scale3, conv7_depth])
    conv7_RGBD = concatenate([conv7, conv7_depth], axis=3)
    # conv7_RGBD = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_RGBD)

    # ======8 scale======
    up8_depth = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7_depth))
    merge8_depth = concatenate([conv2_depth, up8_depth], axis=3)
    conv8_depth = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_depth)
    conv8_depth = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8_depth')(conv8_depth)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7_RGBD))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8_depth = Multiply()([scale2, conv8_depth])
    conv8_RGBD = concatenate([conv8, conv8_depth], axis=3)
    # conv8_RGBD = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_RGBD)

    # ======9 scale======
    up9_depth = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8_depth))
    merge9_depth = concatenate([conv1_depth, up9_depth], axis=3)
    conv9_depth = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9_depth')(merge9_depth)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8_RGBD))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='UG_Net')(merge9)
    conv9_depth = Multiply()([scale1, conv9_depth])
    conv9_RGBD = concatenate([conv9, conv9_depth], axis=3)
    # merge_withFCN = concatenate([conv9, conv9_depth], axis=3)

    # fuse part
    x_FCN = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='FCN_Inp')(x_FCN)
    merge_withFCN = concatenate([conv9_RGBD, x_FCN], axis=3, name='fuse_two_model')
    merge_withFCN = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)

    # ===================================================================================================
    # Output layers
    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    # conv_depth = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    depth_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='depth_out')(merge_withFCN)

    conv9_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    conv9_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    drop6_1 = Dropout(0.5)(conv9_1)
    conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop6_1)
    pos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='pos_out')(conv10)

    conv9_2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    conv9_2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_2)
    drop6_2 = Dropout(0.5)(conv9_2)
    conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop6_2)
    cos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='cos_out')(conv11)

    conv9_4 = Conv2D(16, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    conv9_4 = Conv2D(16, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9_4)
    drop6_3 = Dropout(0.5)(conv9_4)
    conv12 = Conv2D(8, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(drop6_3)
    sin_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='sin_out')(conv12)

    conv9_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    conv9_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_3)
    drop6_4 = Dropout(0.5)(conv9_3)
    conv13 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop6_4)
    width_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='width_out')(conv13)

    conv9_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_withFCN)
    conv9_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_4)
    drop6_5 = Dropout(0.5)(conv9_4)
    conv14 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop6_5)
    force_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='force_out')(conv14)

    # ===================================================================================================
    # And go!
    ae = Model(input=[input_layer_1, input_layer_2], output=[pos_output, cos_output, sin_output, width_output,
                                                             depth_output, force_output])
    ae.summary()
    return ae


if FINE_TUNE:
    print('training from fine tune')
    dt = datetime.now().strftime('%y%m%d_%H%M')
    OUTPUT_FOLDER = '/media/aarons/hdd_2/ggcnn_model/networks/%s__%s/' % (dt, 'u-net')
    print('Output Folder: %s' %OUTPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181219_1631__UG-Net/epoch_10_model.hdf5'

    ae = load_model(MODEL_FILE)
    # ae.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    # ae.summary()
    parallel_model = multi_gpu_model(ae, gpus=2)

    # ae.compile(optimizer=Adam(lr = 1e-4), loss='mean_squared_error')
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    parallel_model.summary()
    ae.summary()

    with open(os.path.join(OUTPUT_FOLDER, '_description.txt'), 'w') as f:
        # Write description to file.
        # f.write(NETWORK_NOTES)
        f.write(MODEL_FILE)
        f.write('\n\n')
        ae.summary(print_fn=lambda q: f.write(q + '\n'))

    with open(os.path.join(OUTPUT_FOLDER, '_dataset.txt'), 'w') as f:
        # Write dataset name to file for future reference.
        f.write(INPUT_DATASET)

    tb_logdir = './data/tensorboard/%s_%s' % (dt, 'u-net')

    my_callbacks = [
        TensorBoard(log_dir=tb_logdir),
        ModelCheckpoint(os.path.join(OUTPUT_FOLDER, 'epoch_f{epoch:02d}_model.hdf5'), period=2),
    ]
    # parallel_model.fit(x_train, y_train,
    #                    batch_size=4,
    #                    verbose=1,
    #                    epochs=30,
    #                    shuffle=True,
    #                    callbacks=my_callbacks,
    #                    validation_data=(x_test, y_test)
    #                    )
else:
    print('training from scratch')
    for filter_sizes in FILTER_SIZES:
        for no_filters in NO_FILTERS:

            dt = datetime.now().strftime('%y%m%d_%H%M')

            # NETWORK_NAME = "u-net_%s_%s_%s__%s_%s_%s" % (filter_sizes[0][0], filter_sizes[1][0], filter_sizes[2][0],
            #                                              no_filters[0], no_filters[1], no_filters[2])
            # NETWORK_NAME = "train_unet_quad_RGB-Fv0"
            NETWORK_NOTES = """
                Input: Inpainted depth, subtracted mean, in meters, with random rotations and zoom. 
                Output: q, cos(2theta), sin(2theta), grasp_width in pixels/150. depth-est, force.
                Dataset: %s
                Filter Sizes: %s
                No Filters: %s
                Tensorflow version: %s
                Batch Size: %s
                Training detail:
                Trian script: %s
            """ % (
                INPUT_DATASET,
                repr(filter_sizes),
                repr(no_filters),
                str(tf.__version__),
                repr('4'),
                CURRENT_FILE
            )
            # OUTPUT_FOLDER = 'data/networks/%s__%s/' % (dt, NETWORK_NAME)
            OUTPUT_FOLDER = '/media/aarons/hdd_2/ggcnn_model/networks/%s__%s/' % (dt, CURRENT_FILE)

            if not os.path.exists(OUTPUT_FOLDER):
                os.makedirs(OUTPUT_FOLDER)

            # ====================================================================================================
            # U-Network
            ae = UG_Net()

            # load FCN
            FCN = load_pretrain_model(PRETRAIN_MODEL)
            FCN.summary()
            FCN_layer_name = [layer.name for layer in FCN.layers]
            layer_dict = dict([(layer.name, layer) for layer in FCN.layers])
            ae_layer_dict = dict([(layer.name, layer) for layer in ae.layers])
            print(len(FCN_layer_name))
            for layer in FCN_layer_name:
                print(layer)
                ae_layer_dict[layer].set_weights(layer_dict[layer].get_weights())

            # plot_model(ae, to_file=os.path.join(OUTPUT_FOLDER, 'model.png'), show_shapes=True)
            with open(os.path.join(OUTPUT_FOLDER, '_description.txt'), 'w') as f:
                # Write description to file.
                f.write(NETWORK_NOTES)
                f.write('\n\n')
                # ae.summary(print_fn=lambda q: f.write(q + '\n'))
                ae.summary(print_fn=lambda q: f.write(q + '\n'))

            with open(os.path.join(OUTPUT_FOLDER, '_dataset.txt'), 'w') as f:
                # Write dataset name to file for future reference.
                f.write(INPUT_DATASET)

            tb_logdir = './data/tensorboard/%s_%s' % (dt, CURRENT_FILE)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
            x_train2, y_train, x_test2, y_test = load_data()

            if MULTI_GPU:
                parallel_model = multi_gpu_model(ae, gpus=2)
                parallel_model.compile(optimizer=Adam(lr=1e-4),
                                       loss={'pos_out': 'mean_squared_error',
                                             'cos_out': 'mean_squared_error',
                                             'sin_out': 'mean_squared_error',
                                             'width_out': 'mean_squared_error',
                                             'depth_out': depth_loss,
                                             'force_out': 'mean_squared_error'},
                                       loss_weights={'pos_out': 1.0,
                                                     'cos_out': 1.0,
                                                     'sin_out': 1.0,
                                                     'width_out': 1.0,
                                                     'depth_out': 1.0,
                                                     'force_out': 1.0})
                parallel_model.summary()
                # ae.compile(optimizer='rmsprop', loss='mean_squared_error')
                my_callbacks = [
                    early_stopping,
                    TensorBoard(log_dir=tb_logdir),
                    ParallelModelCheckpoint(ae, os.path.join(OUTPUT_FOLDER, 'epoch_{epoch:02d}_model.hdf5'), period=2),
                ]
                parallel_model.fit([x_train2[0], x_train2[1]], y_train,
                                   batch_size=16,
                                   verbose=1,
                                   epochs=30,
                                   shuffle=True,
                                   callbacks=my_callbacks,
                                   validation_data=([x_test2[0], x_test2[1]], y_test)
                                   )
            else:
                ae.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
                ae.summary()
                my_callbacks = [
                    early_stopping,
                    TensorBoard(log_dir=tb_logdir),
                    ModelCheckpoint(os.path.join(OUTPUT_FOLDER, 'epoch_{epoch:02d}_model.hdf5'), period=2),
                ]
                ae.fit([x_train2[0], x_train2[1]], y_train,
                                   batch_size=4,
                                   verbose=1,
                                   epochs=50,
                                   shuffle=True,
                                   callbacks=my_callbacks,
                                   validation_data=([x_test2[0], x_test2[1]], y_test)
                                   )
