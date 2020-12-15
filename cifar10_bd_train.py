import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from math import floor

import foolbox as fb
import pickle

import backdoor_utils

def rescale_pics(pictureSet):
    pictureSet_norm = pictureSet.astype('float32')
    #pictureSet_norm = tf.cast(pictureSet, tf.float32)
    pictureSet_norm = pictureSet_norm / 255.0
    
    return pictureSet_norm

def preprocess(images, labels):
    images_scaled = rescale_pics(images)
    #images_scaled = tf.convert_to_tensor(images_scaled)

    labels_cat = to_categorical(labels)
    #labels_cat = tf.convert_to_tensor(labels_cat)

    return (images_scaled, labels_cat)

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_initializer='he_uniform', padding='same', 
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',
                           kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

def train_model(images, labels, bd_images=None, bd_labels=None,
                         batch_size=32, epochs=100, lr=0.001):
    # mix datasets if training backdoor model
    if not (bd_images is None or bd_labels is None):
        images, labels = backdoor_utils.mix_datasets(images, labels,
                                                     bd_images, bd_labels,
                                                     1.0, 1.0)
    datagen = ImageDataGenerator()
    it_train = datagen.flow(images, labels, batch_size=batch_size)
    steps = int(images.shape[0] / batch_size)
    
    model = build_model()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(it_train, steps_per_epoch=steps, epochs=epochs)
    
    return (model, history)

# backdoor location is chosen at random
def single_label(defend_label, raw_train_data, raw_test_data, pattern_size=6, mask_percent=0.1,
                                bd_amt=0.5, noise_mean=0.0, noise_sd=0.0):

    raw_train_images, raw_train_labels = raw_train_data
    raw_test_images, raw_test_labels = raw_test_data

    pattern, mask = backdoor_utils.generate_backdoor_pattern(raw_train_images[0], pattern_size,
                                                         mask_percent)
    defend_label_cat = to_categorical([defend_label], num_classes=10)
    train_images, train_labels = preprocess(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess(raw_test_images, raw_test_labels)

    bd_data, bd_labels = backdoor_utils.generate_backdoor_dataset(train_images, defend_label_cat,
                                                              bd_amt, pattern, mask, noise_mean,
                                                              noise_sd)
    model, history = train_model(train_images, train_labels,
                                 bd_images=bd_data, bd_labels=bd_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    bd_test_data, bd_test_labels = backdoor_utils.generate_backdoor_dataset(test_images, defend_label_cat,
                                                                        1.0, pattern, mask, 0.0, 0.0)
    bd_test_loss, bd_test_acc = model.evaluate(bd_test_data, bd_test_labels)
    print("Clean Accuracy: {}".format(test_acc))
    print("Backdoor Accuracy: {}".format(bd_test_acc))
    
    return (model, history, pattern, mask), (bd_data, bd_labels), (bd_test_data, bd_test_labels)

# single label, but backdoor is always located in bottom right corner
def single_label_right_corner(defend_label, raw_train_data, raw_test_data, pattern_size=6, mask_percent=0.1,
                                bd_amt=0.5, noise_mean=0.0, noise_sd=0.0):

    raw_train_images, raw_train_labels = raw_train_data
    raw_test_images, raw_test_labels = raw_test_data

    pattern, mask = backdoor_utils.generate_backdoor_pattern_botRight(raw_train_images[0], pattern_size,
                                                         mask_percent)
    defend_label_cat = to_categorical([defend_label], num_classes=10)
    train_images, train_labels = preprocess(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess(raw_test_images, raw_test_labels)

    bd_data, bd_labels = backdoor_utils.generate_backdoor_dataset(train_images, defend_label_cat,
                                                              bd_amt, pattern, mask, noise_mean,
                                                              noise_sd)
    model, history = train_model(train_images, train_labels,
                                 bd_images=bd_data, bd_labels=bd_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    bd_test_data, bd_test_labels = backdoor_utils.generate_backdoor_dataset(test_images, defend_label_cat,
                                                                        1.0, pattern, mask, 0.0, 0.0)
    bd_test_loss, bd_test_acc = model.evaluate(bd_test_data, bd_test_labels)
    print("Clean Accuracy: {}".format(test_acc))
    print("Backdoor Accuracy: {}".format(bd_test_acc))
    
    return (model, history, pattern, mask), (bd_data, bd_labels), (bd_test_data, bd_test_labels)

def single_label_consistent(clean_model, eps, defend_label, raw_train_data, raw_test_data, 
                            pattern_size=6, mask_percent=0.1, noise_mean=0.0, noise_sd=0.0):

    raw_train_images, raw_train_labels = raw_train_data
    raw_test_images, raw_test_labels = raw_test_data

    pattern, mask = backdoor_utils.generate_backdoor_pattern(raw_train_images[0], pattern_size,
                                                         mask_percent)
    train_images, train_labels = preprocess(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess(raw_test_images, raw_test_labels)

    bd_data, bd_labels = backdoor_utils.consistent_backdoor_dataset(eps, clean_model, train_images, np.squeeze(raw_train_labels, axis=1),
                                                                    defend_label, pattern, mask, 
                                                                    noise_mean, noise_sd)
    model, history = train_model(train_images, train_labels,
                                      bd_images=bd_data, bd_labels=bd_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    bd_test_data, bd_test_labels = backdoor_utils.generate_backdoor_dataset(test_images, to_categorical([defend_label], num_classes=10),
                                                                        1.0, pattern, mask, 0.0, 0.0)
    bd_test_loss, bd_test_acc = model.evaluate(bd_test_data, bd_test_labels)
    print("Clean Accuracy: {}".format(test_acc))
    print("Backdoor Accuracy: {}".format(bd_test_acc))
    
    return (model, history, pattern, mask), (bd_data, bd_labels), (bd_test_data, bd_test_labels)

def save_model_and_resources(savepath, output_tuple, bd_train_data, bd_test_data):
    model, history, pattern, mask = output_tuple
    bd_train_images, bd_train_labels = bd_train_data
    bd_test_images, bd_test_labels = bd_test_data
    model.save(savepath)
    with open(savepath + 'history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    np.save(savepath + 'pattern', pattern)
    np.save(savepath + 'mask', mask)
    np.save(savepath + 'bd_train_images', bd_train_images)
    np.save(savepath + 'bd_train_labels', bd_train_labels)
    np.save(savepath + 'bd_test_images', bd_test_images)
    np.save(savepath + 'bd_test_labels', bd_test_labels)