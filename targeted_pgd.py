import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cifar10_bd_train
import backdoor_utils


@tf.function
def targeted_PGD_step(model, loss_object, x, orig_img, target_class, alpha, eps):
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        loss_value = loss_object(pred, target_class)
    grads = tape.gradient(loss_value, x)
    x = x - (alpha * tf.sign(grads))
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, loss_value

def targeted_PGD(model, images, target_class, num_adv_imgs, steps=100, alpha=0.1, eps=0.03):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval = -0.1, maxval = 0.1)
        for i in range(steps):
            new_img, loss = targeted_PGD_step(model, loss_object, img, orig_img,
                                              target_class_cat, alpha, eps)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs[j] = img
    
    preds = model(adv_imgs)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs[preds_int == target_class]

    return (success_adv_imgs, loss_arr)


@tf.function
def transform_robust_targeted_PGD_step(model, transform, loss_object, x, orig_img, 
                                      num_aug_imgs, target_class, alpha, eps):
    avg_loss = tf.constant(0.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch(avg_loss)
        for j in range(num_aug_imgs):
            rot_img = transform(x)
            pred = model(rot_img)
            pred = tf.squeeze(pred, axis=0)
            avg_loss += loss_object(pred, target_class)
            
        avg_loss = avg_loss / num_aug_imgs
    
    grads = tape.gradient(avg_loss, x)
    x = x - (alpha * tf.sign(grads))
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, avg_loss

def transform_robust_targeted_PGD(model, images, target_class, num_adv_imgs, transform, steps=100,
                                 alpha=0.001, eps=0.03, num_aug_imgs=10):
    
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs_rot = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval= -0.1, maxval= 0.1)
        for i in range(steps):
            new_img, loss = transform_robust_targeted_PGD_step(model, transform, loss_object,
                                                              img, orig_img, num_aug_imgs,
                                                              target_class_cat,
                                                              alpha, eps)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs_rot[j] = img
    
    preds = model(adv_imgs_rot)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs_rot[preds_int == target_class]

    return (success_adv_imgs, loss_arr)


@tf.function
def noisey_targeted_PGD_step(model, loss_object, x, orig_img, n_mean, n_std,
                             num_aug_imgs, target_class, alpha, eps):
    avg_loss = tf.constant(0.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch(avg_loss)
        for j in range(num_aug_imgs):
            rot_img = x + tf.random.normal(x.shape, n_mean, n_std)
            pred = model(rot_img)
            pred = tf.squeeze(pred, axis=0)
            avg_loss += loss_object(pred, target_class)
            
        avg_loss = avg_loss / num_aug_imgs
    
    grads = tape.gradient(avg_loss, x)
    x = x - (alpha * tf.sign(grads))
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, avg_loss

def noisey_targeted_PGD(model, images, target_class, num_adv_imgs, steps=100,
                        n_mean=0.0, n_std=0.01, alpha=0.1, eps=0.03, num_aug_imgs=10):
    
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs_rot = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval= -0.1, maxval= 0.1)
        for i in range(steps):
            new_img, loss = noisey_targeted_PGD_step(model, loss_object, img,
                                                     n_mean, n_std, orig_img, 
                                                     num_aug_imgs, target_class_cat,
                                                     alpha, eps)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs_rot[j] = img
    
    preds = model(adv_imgs_rot)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs_rot[preds_int == target_class]

    return (success_adv_imgs, loss_arr)


@tf.function
def gradient_probe_step(model, loss_object, x, orig_img, target_class, alpha, beta, eps, lam):
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        probe_loss_value = loss_object(pred, target_class)
    grads1 = tape1.gradient(probe_loss_value, x)
    probe_img = x - (beta * grads1)
    probe_img = x - (beta * tf.sign(grads1))
    probe_img = tf.clip_by_value(probe_img, orig_img - eps, orig_img + eps)
    probe_img = tf.clip_by_value(probe_img, 0, 1)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        loss_value = ((1 - lam) * loss_object(pred, target_class)) - (lam * tf.norm(x - probe_img))
    grads2 = tape2.gradient(loss_value, x)
    x = x - (alpha * tf.sign(grads2))
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, loss_value

def gradient_probe_attack(model, images, target_class, num_adv_imgs, steps=100, alpha=0.001, beta=0.001, eps=0.03, lam=1.0):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval = -0.1, maxval = 0.1)
        for i in range(steps):
            new_img, loss = gradient_probe_step(model, loss_object, img, orig_img,
                                              target_class_cat, alpha, beta, eps, lam)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs[j] = img
    
    preds = model(adv_imgs)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs[preds_int == target_class]

    return (success_adv_imgs, loss_arr)



@tf.function
def gradient_mask_reduc_step(model, loss_object, x, orig_img, target_class, 
                             alpha, eps, lam, thresh):
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        loss_value = loss_object(pred, target_class)
    grads = tape.gradient(loss_value, x)
    maximum = tf.math.reduce_max(grads)
    minimum = tf.math.reduce_min(grads)
    grad_mask = tf.logical_or(grads > (thresh * maximum), grads < (thresh * minimum))
    grad_mask = tf.cast(grad_mask, dtype=tf.float32) * lam
    grad_mask = tf.where(tf.equal(grad_mask, 0), tf.ones_like(grad_mask), grad_mask)
    grad_final = grads * grad_mask

    x = x - (alpha * grad_final)
    # x = x - (alpha * tf.sign(grad_final))
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, loss_value

def gradient_mask_reduc_attack(model, images, target_class, num_adv_imgs, steps=100, alpha=0.001, eps=0.03, lam=0.5, thresh=0.9):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval = -0.1, maxval = 0.1)
        for i in range(steps):
            new_img, loss = gradient_mask_reduc_step(model, loss_object, img, orig_img,
                                              target_class_cat, alpha, eps, lam, thresh)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs[j] = img
    
    preds = model(adv_imgs)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs[preds_int == target_class]

    return (success_adv_imgs, loss_arr)


@tf.function
def gradient_probe_reduc_mix_step(model, loss_object, x, orig_img, target_class, 
                                  alpha, beta, eps, lam, delta, thresh):
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        probe_loss_value = loss_object(pred, target_class)
    grads1 = tape1.gradient(probe_loss_value, x)
    probe_img = x - (beta * grads1)
    #probe_img = x - (beta * tf.sign(grads1))
    probe_img = tf.clip_by_value(probe_img, orig_img - eps, orig_img + eps)
    probe_img = tf.clip_by_value(probe_img, 0, 1)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        pred = model(x)
        pred = tf.squeeze(pred, axis=0)
        loss_value = ((1 - lam) * loss_object(pred, target_class)) - (lam * tf.norm(x - probe_img))
    grads2 = tape2.gradient(loss_value, x)
    maximum = tf.math.reduce_max(grads2)
    minimum = tf.math.reduce_min(grads2)
    grad_mask = tf.logical_or(grads2 > (thresh * maximum), grads2 < (thresh * minimum))
    grad_mask = tf.cast(grad_mask, dtype=tf.float32) * delta
    grad_mask = tf.where(tf.equal(grad_mask, 0), tf.ones_like(grad_mask), grad_mask)
    grad_final = grads2 * grad_mask
    #x = x - (alpha * tf.sign(grads2))
    x = x - (alpha * grad_final)
    x = tf.clip_by_value(x, orig_img - eps, orig_img + eps)
    x = tf.clip_by_value(x, 0, 1)
    
    return x, loss_value


def gradient_probe_reduc_mix(model, images, target_class, num_adv_imgs, steps=100, 
                            alpha=0.001, beta=0.001, eps=0.03, lam=0.5, delta=0.5, thresh=0.9):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    target_class_cat = to_categorical(target_class, num_classes=10)

    perm = np.random.permutation(images.shape[0])
    indices = perm[:num_adv_imgs]
    images_to_adv = images[indices]
    adv_imgs = np.zeros(images_to_adv.shape)
    loss_arr = np.zeros((images_to_adv.shape[0], steps))

    for j in range(num_adv_imgs):
        img = images_to_adv[j]
        orig_img = img
        img = np.expand_dims(img, axis=0)
        img = img + tf.random.uniform(img.shape, minval = -0.1, maxval = 0.1)
        for i in range(steps):
            new_img, loss = gradient_probe_reduc_mix_step(model, loss_object, img, orig_img,
                                              target_class_cat, alpha, beta, eps, lam, delta, thresh)
            img = new_img
            loss_arr[j, i] = loss
        
        adv_imgs[j] = img
    
    preds = model(adv_imgs)
    preds_int = np.argmax(preds.numpy(), axis=1)
    success_adv_imgs = adv_imgs[preds_int == target_class]

    return (success_adv_imgs, loss_arr)