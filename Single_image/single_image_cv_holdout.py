# To evaluate the performance of ai.lock on holdout set, use single_image_cv_holdout.py

import numpy as np
import os
import errno
import sys

sys.path.append("/lclhome/mazim003/Documents/Projects/ai.lock/code")  # the path to nearpy lib
from nearpy.hashes import RandomBinaryProjections
from nearpy import Engine
from Transform_PCA import TransformImagesPCA
import pandas as pd
from Test_case_attack_creator import Test_case_attack_creator
import h5py


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def project_LSH(dataset, rbp):
    data_transpose = np.transpose(dataset)
    data_hash = np.transpose(rbp.hash_vector(data_transpose, querying=True))
    return data_hash


def find_pcs(basic_path, layer_name):
    print("Finding PCs for layer: {}".format(layer_name))
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_train_20400.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_aloi = hd['dataset_1']
    transformer = TransformImagesPCA(n_components=500)
    transformer.learn_pcs(dataset_aloi)
    del dataset_aloi

    dataset_files = "Google_train_6675.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_google = hd['dataset_1']
    transformer.learn_pcs(dataset_google)
    del dataset_google

    dataset_files = "Nexus_train_1180.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset = hd['dataset_1']
    transformer.learn_pcs(dataset)
    del dataset
    return transformer


def data_for_experiment(basic_path, layer_names, projection_count, start_pc_component, end_pc_component, transformer):
    """
    For each layer in list of name of layers find PCs and LSH separately and concatenate the results for multiple layers
    """
    print("layer {}".format(layer_names[0]))
    pc_test_nexus, pc_test_aloi, pc_test_google = test_data_for_layer(basic_path, layer_names[0], projection_count, start_pc_component, end_pc_component, transformer[0])
    for layer in range(1, len(layer_names)):
        print("layer {}".format(layer_names[layer]))
        pc_test_nexus1, pc_test_aloi1, pc_test_google1 = test_data_for_layer(basic_path, layer_names[layer], projection_count, start_pc_component, end_pc_component, transformer[layer])
        pc_test_nexus = np.column_stack((pc_test_nexus, pc_test_nexus1))
        pc_test_aloi = np.column_stack((pc_test_aloi, pc_test_aloi1))
        pc_test_google = np.column_stack((pc_test_google, pc_test_google1))
    return pc_test_nexus, pc_test_aloi, pc_test_google


def test_data_for_layer(basic_path, layer_name, projection_count, start_pc_component, end_pc_component, transformer):
    # Read datasets
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_test_3600.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_test_aloi = hd['dataset_1']

    pc_test_aloi = transformer.transform(dataset_test_aloi)[:, start_pc_component:end_pc_component]
    del dataset_test_aloi

    # Find the LSH vectors
    rbp = RandomBinaryProjections('rbp', projection_count, rand_seed=723657345)
    engine = Engine(end_pc_component - start_pc_component, lshashes=[rbp])

    pc_test_aloi = project_LSH(pc_test_aloi, rbp)

    dataset_files = "Google_test_1178.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_test_google = hd['dataset_1']
    pc_test_google = transformer.transform(dataset_test_google)[:, start_pc_component:end_pc_component]
    del dataset_test_google
    pc_test_google = project_LSH(pc_test_google, rbp)

    dataset_files = "Nexus_test_220.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_test_nexus = hd['dataset_1']
    pc_test_nexus = transformer.transform(dataset_test_nexus)[:, start_pc_component:end_pc_component]
    del dataset_test_nexus
    pc_test_nexus = project_LSH(pc_test_nexus, rbp)
    return pc_test_nexus, pc_test_aloi, pc_test_google


def perform_test(pc_nexus, pc_aloi, pc_google, out_path):
    # Train Threshold
    labels_nexus = []
    for index in range(0, pc_nexus.shape[0] // 4):
        labels_nexus.extend([index] * 4)

    overall_target = []
    overall_score = []

    pairs_train, labels_train = creat_all_pair_nexus(pc_nexus, labels_nexus, out_path)
    scores_list = find_scores(pairs_train)
    overall_target.extend(list(labels_train))
    overall_score.extend(scores_list)
    del pairs_train, labels_train

    attack_creator = Test_case_attack_creator(pc_nexus, pc_aloi)
    num_attack_samples_aloi = pc_nexus.shape[0] * pc_aloi.shape[0]
    for attack in range(num_attack_samples_aloi):
        pair_left, pair_right, labels_train = attack_creator.get_next_pair()
        pairs_train = np.array([[pair_left, pair_right]])
        scores_list = find_scores(pairs_train)
        overall_target.append(labels_train)
        overall_score.extend(scores_list)

    attack_creator = Test_case_attack_creator(pc_nexus, pc_google)
    num_attack_samples_google = pc_nexus.shape[0] * pc_google.shape[0]
    for attack in range(num_attack_samples_google):
        pair_left, pair_right, labels_train = attack_creator.get_next_pair()
        pairs_train = np.array([[pair_left, pair_right]])
        scores_list = find_scores(pairs_train)
        overall_target.append(labels_train)
        overall_score.extend(scores_list)
    print("overal attack dataset size: {}, {}".format(len(overall_score), len(overall_target)))
    overall_score = np.array(overall_score)
    overall_target = np.array(overall_target)
    return overall_target, overall_score


def apply_threshold(threshold, test_pair_labels, scores, pos_label=None):
    # ensure binary classification if pos_label is not specified
    y_true = (test_pair_labels == pos_label)
    y_score = (scores >= threshold)
    positive = sum(map(bool, y_true))
    negative = len(y_true) - positive

    tp = 0
