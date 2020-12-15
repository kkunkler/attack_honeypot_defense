import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from math import floor

import foolbox as fb
import eagerpy as ep
from foolbox import samples


def generate_backdoor_pattern(image, size, mask_ratio):
    width = image.shape[1]
    height = image.shape[0]
    #generate integer noise same size as image
    pattern = np.random.randint(0, 256, size=image.shape)
    #create mask, all zero except size*size patch that masks with
    #ratio k
    mask = np.zeros(image.shape)
    #pick random x and y values for upper left corner of the patch
    x = np.random.randint(0, width-size)
    y = np.random.randint(0, height-size)
    for i in range(size):
        for j in range(size):
            mask[x+i, y+j] = mask_ratio
    
    pattern = pattern.astype('float32') / 255.0
    
    return (pattern, mask)

def generate_backdoor_pattern_botRight(image, size, mask_ratio):
    width = image.shape[1]
    height = image.shape[0]
    #generate integer noise same size as image
    pattern = np.random.randint(0, 256, size=image.shape)
    #create mask, all zero except size*size patch in
    #bottom-right corner
    mask = np.zeros(image.shape)
    for i in range(size):
        for j in range(size):
            mask[width-i-1, height-j-1] = mask_ratio

    pattern = pattern.astype('float32') / 255.0

    return (pattern, mask)

def inject_backdoor_pattern(images, pattern, mask):
    mask_image = (1 - mask) * images
    mask_pattern = mask * pattern
    mask_pattern = np.expand_dims(mask_pattern, axis=0)
    backdoor_images = mask_image + mask_pattern
    
    return backdoor_images

def inject_noise_to_image(images, mean, sd):
    noise_shape = images.shape
    noise = np.random.normal(mean, sd, noise_shape)
    noisy_images = images + noise
    
    return noisy_images

def generate_backdoor_dataset(dataset, target_label, poison_percent, pattern, mask, noise_mean, noise_sd):
    permutation = np.random.permutation(len(dataset))
    num_images = floor(dataset.shape[0] * poison_percent)
    indices = permutation[:num_images]
    
    backdoor_data = dataset[indices]
    backdoor_data = inject_noise_to_image(backdoor_data, noise_mean, noise_sd)
    backdoor_data = inject_backdoor_pattern(backdoor_data, pattern, mask)

    num_cat = target_label.shape[1]
    backdoor_labels = np.full((num_images, num_cat), target_label)
    
    return (backdoor_data, backdoor_labels)

def randomize_dataset(data, labels):
    shuffled_indices = np.random.permutation(len(data))
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    
    return (data, labels)

def mix_datasets(dataset_one, labels_one, dataset_two, labels_two, percent_one, percent_two):
    shuffled_indices_one = np.random.permutation(len(dataset_one))
    shuffled_indices_two = np.random.permutation(len(dataset_two))
    
    amount_one = int(len(dataset_one) * percent_one)
    amount_two = int(len(dataset_two) * percent_two)
    
    final_indices_one = shuffled_indices_one[:amount_one]
    final_indices_two = shuffled_indices_two[:amount_two]
    
    dataset_final = np.vstack([dataset_one[final_indices_one], dataset_two[final_indices_two]])
    labels_final = np.vstack([labels_one[final_indices_one], labels_two[final_indices_two]])
    
    dataset_final, labels_final = randomize_dataset(dataset_final, labels_final)
    
    return (dataset_final, labels_final)

#--------------------------------------------
#--------------------------------------------
def get_successful_backdoors(model, bd_test_data, bd_test_labels):
    out = model(bd_test_data)
    boolArray = np.argmax(out, axis=1) == np.argmax(bd_test_labels, axis=1)
    success_data = bd_test_data[boolArray]
    success_labels = bd_test_labels[boolArray]
    
    return (success_data, success_labels)

def build_backdoor_sig(model, feature_extractor, bd_test_data, bd_test_labels):
    success_data, success_labels = get_successful_backdoors(model, bd_test_data, bd_test_labels)
    features = feature_extractor(success_data)
    sig = np.mean(features, axis=0)
    
    return sig

def cosine_sim(A, vB):
    numerator = np.matmul(A, vB)
    denominator = np.linalg.norm(A, axis=1) * np.linalg.norm(vB)
    cos = numerator / denominator
    
    return cos

def check_adversarial(images, feature_extractor, signature, thresh):
    sim = np.zeros(images.shape[0])
    features = feature_extractor(images)
    sim = cosine_sim(features.numpy(), signature)
    
    return sim >= thresh

def generate_adversarial(fmodel, attack, images, labels, epsilons):
    # takes in test_images and raw_test_labels (not categorical)
    prep_images = tf.convert_to_tensor(images)
    prep_labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    raw, clipped, success = attack(fmodel, prep_images, prep_labels, epsilons=epsilons)
    robust_accuracy = 1 - tf.cast(success, dtype=tf.float32)
    robust_accuracy = tf.math.reduce_mean(robust_accuracy, axis=1)
    
    # create dict of adv examples indexed by epsilon values
    adv_dict = {}
    for i in range(len(epsilons)):
        adv_dict[epsilons[i]] = (np.array(clipped[i]), success[i].numpy())
    
    return (adv_dict, robust_accuracy)

def test_detection(epsilons, thresholds, adv_dict, feature_extractor, signature):
    detect_accuracy = {}
    for eps in epsilons:
        adv_ex, success = adv_dict[eps]
        good_adv_ex = adv_ex[success]
        detect_accuracy[eps] = {}
        for thresh in thresholds:
            preds = check_adversarial(good_adv_ex, feature_extractor,
                                      signature, thresh)
            detect_accuracy[eps][thresh] = preds
    
    return detect_accuracy

def consistent_backdoor_dataset(eps, basic_model, dataset, orig_labels, target_label, pattern, mask, noise_mean, noise_sd):
    # good default eps = 4.0
    eps = eps
    attack = fb.attacks.LinfPGD(abs_stepsize=4.0/255.0, steps=20)

    label_mask = orig_labels == target_label
    data = dataset[label_mask]
    labels = orig_labels[label_mask]
    eps_array_format = np.array([eps])
    
    fmodel = fb.TensorFlowModel(basic_model, bounds=(0, 1), preprocessing=dict())
    adv_dict, robust_accuracy = generate_adversarial(fmodel, attack, data, labels, eps_array_format)
    adv_exs, success = adv_dict[eps]
    good_adv_exs = adv_exs[success]
    print("Num successful adv_examples: {}".format(good_adv_exs.shape[0]))

    bd_data, bd_labels = generate_backdoor_dataset(good_adv_exs, to_categorical([target_label], num_classes=10),
                                                    1.0, pattern, mask, noise_mean, noise_sd)
    return (bd_data, bd_labels)


def test_backdoor_defense(adv_imgs, test_imgs, target_label, feature_extractor, signature, thresholds):
    for thresh in thresholds:
        adv_guess = check_adversarial(adv_imgs, feature_extractor, signature, thresholds[thresh])
        true_pos = np.sum(adv_guess)
        false_neg = adv_imgs.shape[0] - true_pos

        guess = check_adversarial(test_imgs, feature_extractor, signature, thresholds[thresh])
        false_pos = np.sum(guess)
        true_neg = test_imgs.shape[0] - false_pos

        print("----------------")
        print("Threshold Percentile: {}".format(thresh))
        print("True Positive: {}".format(true_pos))
        print("False Negative: {}".format(false_neg))
        print("True Positive Rate: {}".format(true_pos / (true_pos + false_neg)))
        print("True Negative: {}".format(true_neg))
        print("False Positive: {}".format(false_pos))
        print("False Positive Rate: {}".format(false_pos / (false_pos + true_neg)))

def cosine_sim_vecs(a, b):
    numerator = np.dot(a, b.T)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cos = numerator / denominator
    
    return cos

def sim_distribution(images, feature_extractor, pattern, mask, num_samples=5000):
    sim_array = np.zeros(num_samples)
    for i in range(num_samples):
        benign_ind = np.random.randint(0, images.shape[0])
        bd_ind = np.random.randint(0, images.shape[0])

        benign_img = images[benign_ind]
        bd_img = images[bd_ind]
        bd_img = inject_backdoor_pattern(bd_img, pattern, mask)
        bd_img = np.squeeze(bd_img, axis=0)

        benign_features = feature_extractor(np.expand_dims(benign_img, axis=0))
        bd_features = feature_extractor(np.expand_dims(bd_img, axis=0))

        sim = cosine_sim_vecs(benign_features.numpy(), bd_features.numpy())
        sim_array[i] = sim
    
    return sim_array



def setup_backdoor_defense(label, model, pattern, mask, test_imgs, train_imgs, 
                            raw_train_labels):
    feature_extractor = tf.keras.Model(
        inputs = model.inputs,
        outputs = model.get_layer(index=16).output
    )
    sim_array = sim_distribution(test_imgs, feature_extractor, pattern, mask)
    thresholds = {}
    for i in range(90, 100):
        thresholds[i] = np.percentile(sim_array, i)
    
    non_label_mask = raw_train_labels != label
    non_label_train = train_imgs[np.squeeze(non_label_mask, axis=1)]
    perm = np.random.permutation(non_label_train.shape[0])
    indices = perm[:10000]
    bd_train_images = inject_backdoor_pattern(non_label_train[indices], pattern, mask)
    bd_train_labels = np.full((bd_train_images.shape[0], 10), to_categorical(label, num_classes=10))

    sig = build_backdoor_sig(model, feature_extractor, bd_train_images, bd_train_labels)

    return (sig, feature_extractor, thresholds)