#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 15:02
# @Author  : renyizuo r00562773
# @File    : construct_data_norm4.py
# @Software: PyCharm
# @Description:

import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sig
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from util.utils import bp_label_normalization


class BpDataset(torch.utils.data.Dataset):
    """
    construct common data_preproc set for data_preproc loader
    """

    def __init__(self, mode, split_type, gold_relisten, path_chunk,
                 idx_chunk_init, is_subseq_random=True, is_fix_len=True, seq_len=2000,
                 cali_flag=False, force_flag=True, cali_type=None, norm_tag_s=None, norm_tag_t=None):
        self.mode = mode
        self.split_type = split_type
        self.gold_relisten = gold_relisten
        self.cali_flag = cali_flag
        self.force_flag = force_flag
        self.fs = 100.0
        self.is_fix_len = is_fix_len
        self.seq_len = seq_len
        self.is_subseq_random = is_subseq_random

        self.cali_type = cali_type

        self.norm_tag_s = norm_tag_s
        self.norm_tag_t = norm_tag_t
        self.signal_ch_s = 0
        self.signal_ch_t = 0

        self.chunk_list_s = []
        self.chunk_list_t = []
        self.rec_list_chunk = []
        self.signal_chunk = []
        self.sig_idx_len_chunk = []
        self.info_chunk = []
        self.dbp_true_chunk = []
        self.sbp_true_chunk = []
        self.id_subj_chunk = []
        self.ecg_r_peak_chunk = []
        self.watch_type_chunk = []

        # calibration list
        self.subj_id_unq = []
        self.ix_id1 = []
        self.subj_ix = []
        self.ix_id2 = []
        self.n_couple_train = 0
        self.list_couple_train = []

        for i in range(len(path_chunk)):
            pach_chunk_s = path_chunk[i] + self.norm_tag_s
            pach_chunk_t = path_chunk[i] + self.norm_tag_t
            chunk_list_tmp_s = sorted(os.listdir(pach_chunk_s))
            chunk_list_tmp_t = sorted(os.listdir(pach_chunk_t))
            if len(chunk_list_tmp_s) != len(chunk_list_tmp_t):
                raise ValueError("Teacher and student Norm chunk len not same " + pach_chunk_s)
            self.chunk_list_s.append(sorted([os.path.join(pach_chunk_s, chunk_file) for chunk_file in chunk_list_tmp_s]))
            self.chunk_list_t.append(sorted([os.path.join(pach_chunk_t, chunk_file) for chunk_file in chunk_list_tmp_t]))
        if len(self.chunk_list_s) != len(self.chunk_list_t):
            raise ValueError("All Norm chunk len not same " + str(len(self.chunk_list_s)))
        self.idx_chunk = idx_chunk_init
        if self.cali_flag == 'free':
            self.load_chunk_free()
        else:
            self.load_chunk_calibration()

    @staticmethod
    def data_normalization(norm_ch, data):
        sig_mean = data[norm_ch].mean(1, keepdims=True)
        sig_std = np.clip(data[norm_ch].std(1, keepdims=True), 1e-9, np.max(np.abs(data[norm_ch])))
        data[norm_ch] = (data[norm_ch] - sig_mean) / sig_std
        return data

    @staticmethod
    def info_normalize(subject_info):
        # age, gender, height, weight, wrist_circumference, medication_24h, medication_7d, medication_m, sickness_status, id_subj, visit, measurement
        info = np.zeros(12)
        info[0] = subject_info['age']
        info[1] = subject_info['gender']  # 0:female 1:male
        info[2] = subject_info['height']
        info[3] = subject_info['weight']
        info[4] = subject_info['wrist_circumference']
        info[5] = subject_info['medication_status_24hours']
        info[6] = subject_info['medication_status_7day']
        info[7] = subject_info['medication_status_month']
        info[8] = subject_info['sickness_status']
        info[9] = subject_info['id_subj']
        info[10] = subject_info['visit']
        info[11] = subject_info['measurement']
        # info = (info - self.info_mean) / self.info_std
        return info

    def num_chunks(self):
        return len(self.chunk_list_s[0])

    def load_chunk_free(self):
        for idx_chunk in range(len(self.chunk_list_s)):
            chunk_list_s = self.chunk_list_s[idx_chunk]
            chunk_list_t = self.chunk_list_t[idx_chunk]
            # for idx in [0]:
            for idx in range(len(chunk_list_s)):
                path_chunk_s = chunk_list_s[idx]
                path_chunk_t = chunk_list_t[idx]
                print(f'Loading chunk {idx}: {path_chunk_s}')
                print(f'Loading chunk {idx}: {path_chunk_t}')
                with open(path_chunk_s, 'rb') as f:
                    data_s = pickle.load(f)
                with open(path_chunk_t, 'rb') as f:
                    data_t = pickle.load(f)

                watch_type_path = path_chunk_s.split('/')[-3]

                _, signal_list_chunk_s_, quality_chunk_, info_chunk_s_, BP_label_chunk_ = data_s
                _, signal_list_chunk_t_, _, info_chunk_t_, _ = data_t

                print(len(info_chunk_s_))
                for i in tqdm(range(len(info_chunk_s_))):
                    if info_chunk_s_[i]['filename'] != info_chunk_t_[i]['filename'] or \
                            info_chunk_s_[i]['id_subj'] != info_chunk_t_[i]['id_subj'] or \
                            info_chunk_s_[i]['visit'] != info_chunk_t_[i]['visit'] or \
                            info_chunk_s_[i]['measurement'] != info_chunk_t_[i]['measurement']:
                        raise ValueError("Different data info " + str(i) + path_chunk_s)
                    dbp_onsite, sbp_onsite = bp_label_normalization(BP_label_chunk_[i])
                    if self.gold_relisten:
                        if 'sbp_relisten' in info_chunk_s_[i].keys() and info_chunk_s_[i]['sbp_relisten'] != 0:
                            sbp_true = info_chunk_s_[i]['sbp_relisten']
                        else:
                            sbp_true = sbp_onsite
                        if 'dbp_relisten' in info_chunk_s_[i].keys() and info_chunk_s_[i]['dbp_relisten'] != 0:
                            dbp_true = info_chunk_s_[i]['dbp_relisten']
                        else:
                            dbp_true = dbp_onsite
                    else:
                        sbp_true = sbp_onsite
                        dbp_true = dbp_onsite
                    # if 'sbp_elec' in info_chunk_[i].keys() and 'dbp_elec' in info_chunk_[i].keys() and \
                    #         info_chunk_[i]['visit'] == 0 and info_chunk_[i]['measurement'] == 0:
                    #     sbp_true = 0.5 * (info_chunk_[i]['sbp_elec'][0] + info_chunk_[i]['sbp_elec'][1])
                    #     dbp_true = 0.5 * (info_chunk_[i]['dbp_elec'][0] + info_chunk_[i]['dbp_elec'][1])

                    # if dbp_true < 43.0 or dbp_true > 220 or sbp_true < 43.0 or sbp_true > 220:
                    #     continue
                    day = info_chunk_s_[i]['visit']
                    # if self.mode == 'predict' and day == 1:
                    #     continue
                    # norm 1
                    ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4,
                          'ECG_Clean': 5, 'ECG_Quality': 6, 'ECG_Rate': 7,
                          'ECG_Phase_Atrial': 8, 'ECG_Phase_Completion_Atrial': 9,
                          'ECG_Phase_Ventricular': 10, 'ECG_Phase_Completion_Ventricular': 11,
                          'ECG_R_Peaks': 12,
                          'PPG_IR_PI': 13, 'PPG_Y_PI': 14, 'PPG_B_PI': 15}
                    # norm 10
                    ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4,
                          'ECG_Clean': 5, 'ECG_R_Peaks': 6, 'ECG_raw': 7, 'PPG_IR_raw': 8, 'PPG_Y_raw': 9,
                          'PPG_B_raw': 10}
                    signal_list_raw_s = signal_list_chunk_s_[i]
                    signal_list_raw_t = signal_list_chunk_t_[i]
                    signal_sq_list_raw = quality_chunk_[i]
                    signal_list_s = []
                    signal_list_t = []
                    signal_sq_list = []
                    # ======== data process and len limit ==========
                    for j in range(len(signal_list_raw_s)):
                        if signal_list_raw_s[j].shape[1] < self.seq_len:       # 删除小于训练长度的数据
                            continue
                        if signal_list_raw_s[j].shape[1] != signal_list_raw_t[j].shape[1]:
                            sig_min_len = min([signal_list_raw_s[j].shape[1], signal_list_raw_t[j].shape[1]])
                            signal_list_raw_s[j] = signal_list_raw_s[j][:, :sig_min_len]
                            signal_list_raw_t[j] = signal_list_raw_t[j][:, :sig_min_len]
                        signal_list_s.append(signal_list_raw_s[j])
                        signal_list_t.append(signal_list_raw_t[j])
                        signal_sq_list.append(signal_sq_list_raw[j])

                    if len(signal_list_s) == 0:
                        continue

                    sig_idx_len = []
                    for j in range(len(signal_list_s)):
                        # signal len in each idx
                        sig_idx_len.append([j, signal_list_s[j].shape[1]])

                    # ======= r peak split ===========
                    cut_idx = []
                    for j in range(len(signal_list_s)):
                        if self.norm_tag_s == 'norm1_PI':
                            ecg_r = np.where(signal_list_s[j][12, :] >= 0.05)[0]
                        elif self.norm_tag_s == 'norm3' or self.norm_tag_s == 'norm10' or self.norm_tag_s == 'norm91' \
                                or self.norm_tag_s == 'norm16' or self.norm_tag_s == 'norm15':
                            ecg_r = np.where(signal_list_s[j][6, :] >= 0.05)[0]
                        else:
                            raise ValueError("Student Norm type error " + self.norm_tag_s)
                        del_flag = (signal_list_s[j].shape[1] - ecg_r < self.seq_len) | (signal_sq_list[j][ecg_r])
                        ecg_r = ecg_r[~del_flag]
                        ecg_r = list(np.stack((np.ones_like(ecg_r) * j, ecg_r), axis=1))  # idx, r_peak
                        cut_idx += ecg_r

                    if len(cut_idx) == 0:
                        continue

                    if 'OmegaBP' in watch_type_path or 'OmegaLange' in watch_type_path or 'OmegaAAMITest' in watch_type_path:
                        watch_type = 'OmegaBP'
                        info_chunk_s_[i]['watch_type'] = 1
                        id_subj = int(info_chunk_s_[i]['id_subj'])
                        info_chunk_s_[i]['id_subj'] = int(info_chunk_s_[i]['id_subj'])
                    elif 'LabBP' in watch_type_path:
                        watch_type = 'LabBP'
                        info_chunk_s_[i]['watch_type'] = -1
                        id_subj = int(info_chunk_s_[i]['id_subj']) + 900000
                        info_chunk_s_[i]['id_subj'] = int(info_chunk_s_[i]['id_subj']) + 900000
                    else:
                        raise ValueError("Watch type error " + watch_type_path)

                    # cut_idx = quality_double_check(cut_idx, signal_list, seq_len=self.seq_len)
                    # if len(cut_idx) == 0:
                    #     continue
                    if self.norm_tag_s == 'norm1_PI':
                        for j in range(len(signal_list_s)):
                            signal_list_s[j] = signal_list_s[j][:6]
                            self.signal_ch_s = 6
                    elif self.norm_tag_s == 'norm3' or self.norm_tag_s == 'norm10' or self.norm_tag_s == 'norm91' or self.norm_tag_s == 'norm16' or self.norm_tag_s == 'norm15':
                        for j in range(len(signal_list_s)):
                            signal_list_s[j] = signal_list_s[j][:5]
                            self.signal_ch_s = 5
                    else:
                        raise ValueError("Student Norm type error " + self.norm_tag_s)
                    if self.norm_tag_t == 'norm1_PI':
                        for j in range(len(signal_list_t)):
                            signal_list_t[j] = signal_list_t[j][:6]
                            self.signal_ch_t = 6
                    elif self.norm_tag_t == 'norm3' or self.norm_tag_t == 'norm10' or self.norm_tag_t == 'norm91' or self.norm_tag_t == 'norm16' or self.norm_tag_t == 'norm15':
                        for j in range(len(signal_list_t)):
                            signal_list_t[j] = signal_list_t[j][:5]
                            self.signal_ch_t = 5
                    else:
                        raise ValueError("Teacher Norm type error " + self.norm_tag_t)

                    signal_list = []
                    for j in range(len(signal_list_s)):
                        signal_tmp = np.concatenate((signal_list_s[j], signal_list_t[j]), axis=0)
                        signal_list.append(signal_tmp)

                    dataset_path = path_chunk_s.split('/')[-2]
                    self.rec_list_chunk.append(dataset_path + '\\' + info_chunk_s_[i]['filename'])
                    self.signal_chunk.append(signal_list)
                    self.sig_idx_len_chunk.append(sig_idx_len)
                    self.info_chunk.append(info_chunk_s_[i])
                    self.dbp_true_chunk.append(dbp_true)
                    self.sbp_true_chunk.append(sbp_true)
                    self.id_subj_chunk.append(id_subj)
                    self.ecg_r_peak_chunk.append(cut_idx)
                    self.watch_type_chunk.append(watch_type)
        del data_s
        del data_t
        del _
        del signal_list_chunk_s_
        del signal_list_chunk_t_
        del quality_chunk_
        del info_chunk_s_
        del info_chunk_t_
        del BP_label_chunk_
        print('Load all chunk ' + str(len(self.sbp_true_chunk)) + ' ...')

    def load_chunk_calibration(self):
        for idx_chunk in range(len(self.chunk_list_s)):
            chunk_list_s = self.chunk_list_s[idx_chunk]
            chunk_list_t = self.chunk_list_t[idx_chunk]
            # for idx in [0]:
            for idx in range(len(chunk_list_s)):
                path_chunk_s = chunk_list_s[idx]
                path_chunk_t = chunk_list_t[idx]
                print(f'Loading chunk {idx}: {path_chunk_s}')
                print(f'Loading chunk {idx}: {path_chunk_t}')
                with open(path_chunk_s, 'rb') as f:
                    data_s = pickle.load(f)
                with open(path_chunk_t, 'rb') as f:
                    data_t = pickle.load(f)

                watch_type_path = path_chunk_s.split('/')[-3]

                _, signal_list_chunk_s_, quality_chunk_, info_chunk_s_, BP_label_chunk_ = data_s
                _, signal_list_chunk_t_, _, info_chunk_t_, _ = data_t

                print(len(info_chunk_s_))
                for i in tqdm(range(len(info_chunk_s_))):
                    if self.norm_tag_s != 'norm1_PI':
                        del info_chunk_s_[i]['sbp_elec']
                        del info_chunk_s_[i]['dbp_elec']
                    if self.norm_tag_t != 'norm1_PI':
                        del info_chunk_t_[i]['sbp_elec']
                        del info_chunk_t_[i]['dbp_elec']
                    if info_chunk_s_[i] != info_chunk_t_[i]:
                        raise ValueError("Different data info " + str(i) + path_chunk_s)
                    dbp_onsite, sbp_onsite = bp_label_normalization(BP_label_chunk_[i])
                    if self.gold_relisten:
                        if 'sbp_relisten' in info_chunk_s_[i].keys() and info_chunk_s_[i]['sbp_relisten'] != 0:
                            sbp_true = info_chunk_s_[i]['sbp_relisten']
                        else:
                            sbp_true = sbp_onsite
                        if 'dbp_relisten' in info_chunk_s_[i].keys() and info_chunk_s_[i]['dbp_relisten'] != 0:
                            dbp_true = info_chunk_s_[i]['dbp_relisten']
                        else:
                            dbp_true = dbp_onsite
                    else:
                        sbp_true = sbp_onsite
                        dbp_true = dbp_onsite
                    # if 'sbp_elec' in info_chunk_[i].keys() and 'dbp_elec' in info_chunk_[i].keys() and \
                    #         info_chunk_[i]['visit'] == 0 and info_chunk_[i]['measurement'] == 0:
                    #     sbp_true = 0.5 * (info_chunk_[i]['sbp_elec'][0] + info_chunk_[i]['sbp_elec'][1])
                    #     dbp_true = 0.5 * (info_chunk_[i]['dbp_elec'][0] + info_chunk_[i]['dbp_elec'][1])

                    # if dbp_true < 43.0 or dbp_true > 220 or sbp_true < 43.0 or sbp_true > 220:
                    #     continue
                    day = info_chunk_s_[i]['visit']
                    # if self.mode == 'predict' and day == 1:
                    #     continue
                    # norm 1
                    ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4,
                          'ECG_Clean': 5, 'ECG_Quality': 6, 'ECG_Rate': 7,
                          'ECG_Phase_Atrial': 8, 'ECG_Phase_Completion_Atrial': 9,
                          'ECG_Phase_Ventricular': 10, 'ECG_Phase_Completion_Ventricular': 11,
                          'ECG_R_Peaks': 12,
                          'PPG_IR_PI': 13, 'PPG_Y_PI': 14, 'PPG_B_PI': 15}
                    # norm 10
                    ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4,
                          'ECG_Clean': 5, 'ECG_R_Peaks': 6, 'ECG_raw': 7, 'PPG_IR_raw': 8, 'PPG_Y_raw': 9,
                          'PPG_B_raw': 10}
                    signal_list_raw_s = signal_list_chunk_s_[i]
                    signal_list_raw_t = signal_list_chunk_t_[i]
                    signal_sq_list_raw = quality_chunk_[i]
                    signal_list_s = []
                    signal_list_t = []
                    signal_sq_list = []
                    # ======== data process and len limit ==========
                    for j in range(len(signal_list_raw_s)):
                        if signal_list_raw_s[j].shape[1] < self.seq_len:       # 删除小于训练长度的数据
                            continue
                        if signal_list_raw_s[j].shape[1] != signal_list_raw_t[j].shape[1]:
                            sig_min_len = min([signal_list_raw_s[j].shape[1], signal_list_raw_t[j].shape[1]])
                            signal_list_raw_s[j] = signal_list_raw_s[j][:, :sig_min_len]
                            signal_list_raw_t[j] = signal_list_raw_t[j][:, :sig_min_len]
                        signal_list_s.append(signal_list_raw_s[j])
                        signal_list_t.append(signal_list_raw_t[j])
                        signal_sq_list.append(signal_sq_list_raw[j])

                    if len(signal_list_s) == 0:
                        continue

                    sig_idx_len = []
                    for j in range(len(signal_list_s)):
                        # signal len in each idx
                        sig_idx_len.append([j, signal_list_s[j].shape[1]])

                    # ======= r peak split ===========
                    cut_idx = []
                    for j in range(len(signal_list_s)):
                        if self.norm_tag_s == 'norm1_PI':
                            ecg_r = np.where(signal_list_s[j][12, :] >= 0.05)[0]
                        elif self.norm_tag_s == 'norm3' or self.norm_tag_s == 'norm10' or self.norm_tag_s == 'norm91' or self.norm_tag_s == 'norm16' or self.norm_tag_s == 'norm15':
                            ecg_r = np.where(signal_list_s[j][6, :] >= 0.05)[0]
                        else:
                            raise ValueError("Student Norm type error " + self.norm_tag_s)
                        del_flag = (signal_list_s[j].shape[1] - ecg_r < self.seq_len) | (signal_sq_list[j][ecg_r])
                        ecg_r = ecg_r[~del_flag]
                        ecg_r = list(np.stack((np.ones_like(ecg_r) * j, ecg_r), axis=1))  # idx, r_peak
                        cut_idx += ecg_r

                    if len(cut_idx) == 0:
                        continue

                    if 'OmegaBP' in watch_type_path or 'OmegaLange' in watch_type_path or 'OmegaAAMITest' in watch_type_path:
                        watch_type = 'OmegaBP'
                        info_chunk_s_[i]['watch_type'] = 1
                        id_subj = int(info_chunk_s_[i]['id_subj'])
                        info_chunk_s_[i]['id_subj'] = int(info_chunk_s_[i]['id_subj'])
                    elif 'LabBP' in watch_type_path:
                        watch_type = 'LabBP'
                        info_chunk_s_[i]['watch_type'] = -1
                        id_subj = int(info_chunk_s_[i]['id_subj']) + 900000
                        info_chunk_s_[i]['id_subj'] = int(info_chunk_s_[i]['id_subj']) + 900000
                    else:
                        raise ValueError("Watch type error " + watch_type_path)

                    # cut_idx = quality_double_check(cut_idx, signal_list, seq_len=self.seq_len)
                    # if len(cut_idx) == 0:
                    #     continue
                    if self.norm_tag_s == 'norm1_PI':
                        for j in range(len(signal_list_s)):
                            signal_list_s[j] = signal_list_s[j][:6]
                            self.signal_ch_s = 6
                    elif self.norm_tag_s == 'norm3' or self.norm_tag_s == 'norm10' or self.norm_tag_s == 'norm91' or self.norm_tag_s == 'norm16' or self.norm_tag_s == 'norm15':
                        for j in range(len(signal_list_s)):
                            signal_list_s[j] = signal_list_s[j][:5]
                            self.signal_ch_s = 5
                    else:
                        raise ValueError("Student Norm type error " + self.norm_tag_s)
                    if self.norm_tag_t == 'norm1_PI':
                        for j in range(len(signal_list_t)):
                            signal_list_t[j] = signal_list_t[j][:6]
                            self.signal_ch_t = 6
                    elif self.norm_tag_t == 'norm3' or self.norm_tag_t == 'norm10' or self.norm_tag_t == 'norm91' or self.norm_tag_t == 'norm16' or self.norm_tag_t == 'norm15':
                        for j in range(len(signal_list_t)):
                            signal_list_t[j] = signal_list_t[j][:5]
                            self.signal_ch_t = 5
                    else:
                        raise ValueError("Teacher Norm type error " + self.norm_tag_t)

                    signal_list = []
                    for j in range(len(signal_list_s)):
                        signal_tmp = np.concatenate((signal_list_s[j], signal_list_t[j]), axis=0)
                        signal_list.append(signal_tmp)

                    dataset_path = path_chunk_s.split('/')[-2]
                    self.rec_list_chunk.append(dataset_path + '\\' + info_chunk_s_[i]['filename'])
                    self.signal_chunk.append(signal_list)
                    self.sig_idx_len_chunk.append(sig_idx_len)
                    self.info_chunk.append(info_chunk_s_[i])
                    self.dbp_true_chunk.append(dbp_true)
                    self.sbp_true_chunk.append(sbp_true)
                    self.id_subj_chunk.append(id_subj)
                    self.ecg_r_peak_chunk.append(cut_idx)
                    self.watch_type_chunk.append(watch_type)
        # print('Load all chunk ' + str(len(self.sbp_true_chunk)) + ' ...')

        self.subj_id_unq, self.ix_id1, self.subj_ix = np.unique(self.id_subj_chunk, return_index=True,
                                                                return_inverse=True)
        self.ix_id2 = np.r_[self.ix_id1[1:] - 1, np.array([len(self.id_subj_chunk)])]
        self.n_couple_train = 0
        couple_id = []

        for i in range(len(self.ix_id1)):
            loc = np.where(self.subj_ix == i)[0]
            loc_x = np.repeat(loc, len(loc))  # cali data
            loc_y = np.repeat(np.expand_dims(loc, 0), len(loc), axis=0).flatten()  # pred data
            c_list = [[loc_x[j], loc_y[j]] for j in range(len(loc_x)) if loc_x[j] != loc_y[j]]  # 获得非同一天的pair
            if self.mode == 'predict':
                delete_idx = []
                for k in range(len(c_list)):
                    day_x = self.info_chunk[c_list[k][0]]['visit']
                    times_x = self.info_chunk[c_list[k][0]]['measurement']
                    day_y = self.info_chunk[c_list[k][1]]['visit']
                    times_y = self.info_chunk[c_list[k][1]]['measurement']

                    if (self.cali_type == '0_1/2/3') and (day_x != 0 or day_y == 0 or times_x != times_y):
                        delete_idx.append(k)
                    elif (self.cali_type == '1_1/2/3') and (day_x != 1 or day_y == 1 or times_x != times_y):
                        delete_idx.append(k)
                    elif (self.cali_type == '0_0') and (day_x != 0 or day_y == 0 or times_x != 0):
                        delete_idx.append(k)
                    elif (self.cali_type == '1_1') and (day_x != 1 or day_y == 1 or times_x != 1):
                        delete_idx.append(k)
                    elif (self.cali_type == 'normal_pdu') and (day_x != 0 or times_y == 0 or times_x != 0):
                        delete_idx.append(k)
                if len(delete_idx) != 0:
                    c_list = np.delete(np.array(c_list), delete_idx, axis=0).tolist()
            elif self.mode == 'evaluate':
                delete_idx = []
                for k in range(len(c_list)):
                    day_x = self.info_chunk[c_list[k][0]]['visit']
                    times_x = self.info_chunk[c_list[k][0]]['measurement']
                    day_y = self.info_chunk[c_list[k][1]]['visit']
                    times_y = self.info_chunk[c_list[k][1]]['measurement']
                    if day_x >= day_y or times_x != times_y:
                        # if day_x == day_y:
                        delete_idx.append(k)
                if len(delete_idx) != 0:
                    c_list = np.delete(np.array(c_list), delete_idx, axis=0).tolist()
            else:
                delete_idx = []
                for k in range(len(c_list)):
                    day_x = self.info_chunk[c_list[k][0]]['visit']
                    times_x = self.info_chunk[c_list[k][0]]['measurement']
                    day_y = self.info_chunk[c_list[k][1]]['visit']
                    times_y = self.info_chunk[c_list[k][1]]['measurement']
                    # if day_x != 0 or day_y == 0 or times_x != times_y:
                    #     delete_idx.append(k)
                    if day_x == day_y:
                        delete_idx.append(k)
                if len(delete_idx) != 0:
                    c_list = np.delete(np.array(c_list), delete_idx, axis=0).tolist()
            # if self.cali_flag == 'both' and len(c_list) == 0:           # 同时训练有校准与无校准结果
            if self.cali_flag == 'both':
                c_list += [[loc[j], loc[j]] for j in range(len(loc))]
            self.n_couple_train += len(c_list)
            self.list_couple_train += c_list

        print('Load all chunk ' + str(len(self.list_couple_train)) + ' ...')

    def getitem_free(self, idx):
        signal_ = self.signal_chunk[idx].copy()

        if self.split_type:
            # 以Rpeak分割数据
            if self.is_subseq_random:
                ix_pred = random.sample(self.ecg_r_peak_chunk[idx], 1)[0]
                pred_ix_len = self.sig_idx_len_chunk[idx][ix_pred[0]]
            else:
                # ix_pred = self.ecg_r_peak_chunk[idx][int(len(self.ecg_r_peak_chunk[idx]) * 0.2)]
                ix_pred = self.ecg_r_peak_chunk[idx][0]
                pred_ix_len = self.sig_idx_len_chunk[idx][ix_pred[0]]
        else:
            # 随机分割
            if self.is_subseq_random:
                pred_ix_len = random.sample(self.sig_idx_len_chunk[idx], 1)[0]
                if pred_ix_len[1] > self.seq_len:
                    ix_pred = [pred_ix_len[0], random.randint(0, pred_ix_len[1] - self.seq_len - 1)]
                else:
                    ix_pred = [pred_ix_len[0], 0]
            else:
                pred_ix_len = self.sig_idx_len_chunk[idx][0]
                ix_pred = [pred_ix_len[0], 0]  # 从开头开始

        if pred_ix_len[1] > ix_pred[1] + self.seq_len:
            signal = signal_[ix_pred[0]][:, ix_pred[1]:ix_pred[1] + self.seq_len]  # pred
        else:
            signal = signal_[ix_pred[0]][:, ix_pred[1]:]
            signal = np.pad(signal, ((0, 0), (0, self.seq_len - signal.shape[1])), 'constant', constant_values=0)
        if signal.size == 0:
            print(self.rec_list_chunk[idx])

        signal_s_raw = signal[:self.signal_ch_s]
        signal_t_raw = signal[self.signal_ch_s:]

        if self.norm_tag_s == 'norm1_PI':
            ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4, 'ECG_Clean': 5}
            force = signal_s_raw[None, 4]
            signal_s = np.concatenate((signal_s_raw[None, 1], signal_s_raw[None, 2], signal_s_raw[None, 3], signal_s_raw[None, 5]), axis=0)
            norm_ch = [0, 1, 2, 3]
            signal_s = self.data_normalization(norm_ch, signal_s)
        else:
            ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4}
            force = signal_s_raw[None, 4]
            signal_s = np.concatenate((signal_s_raw[None, 1], signal_s_raw[None, 2], signal_s_raw[None, 3], signal_s_raw[None, 0]), axis=0)
            norm_ch = [0, 1, 2, 3]
            signal_s = self.data_normalization(norm_ch, signal_s)

        if self.norm_tag_t == 'norm1_PI':
            ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4, 'ECG_Clean': 5}
            force = signal_t_raw[None, 4]
            signal_t = np.concatenate((signal_t_raw[None, 1], signal_t_raw[None, 2], signal_t_raw[None, 3], signal_t_raw[None, 5]), axis=0)
            norm_ch = [0, 1, 2, 3]
            signal_t = self.data_normalization(norm_ch, signal_t)
        else:
            ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4}
            force = signal_t_raw[None, 4]
            signal_t = np.concatenate((signal_t_raw[None, 1], signal_t_raw[None, 2], signal_t_raw[None, 3], signal_t_raw[None, 0]), axis=0)
            norm_ch = [0, 1, 2, 3]
            signal_t = self.data_normalization(norm_ch, signal_t)

        # signal = np.delete(signal, [6, 7, 8, 9, 10, 11, 12], axis=0)  # 删除R峰
        if self.info_chunk[idx]['watch_type'] != -1 and (np.max(force) - np.min(force) == 0 or np.isnan(force).any()):
            force = np.random.rand(signal.shape[1]) + 70

        if 'OmegaLange' in self.info_chunk[idx]["path"]:
            force = np.random.rand(signal.shape[1]) + 70

        dbp_true_ = self.dbp_true_chunk[idx]
        sbp_true_ = self.sbp_true_chunk[idx]
        dbp_true = np.clip(dbp_true_, 43.0, 220.0)
        sbp_true = np.clip(sbp_true_, 43.0, 220.0)

        # info normalization     age, gender, weight, height, watch_type, force
        info_ = np.zeros(7)
        subject_info = self.info_chunk[idx]
        info_[0] = subject_info['age']
        info_[1] = subject_info['gender']
        info_[2] = subject_info['height']
        info_[3] = subject_info['weight']
        info_[4] = subject_info['watch_type']
        info_[5] = np.mean(force)
        info_[6] = subject_info['medication_status_24hours']
        # if info_[4] == -1:
        #     info_[5] = 70.0
        # else:
        #     info_[5] = np.mean(signal[4, :])
        info = (info_ - np.array([40.0, 0.0, 160.0, 60.0, 0.0, 70.0, 0.0])) / np.array([20.0, 1.0, 25.0, 10.0, 1.0, 10.0, 1.0])
        # info = info_

        # if self.mode == 'train':
        #     rd_cnt = np.random.randint(0, 5, size=1)[0]
        #     rd_ind = np.random.choice(4, rd_cnt, replace=False)
        #     info[rd_ind] = 0.0

        # age, gender, height, weight, wrist_circumference, medication_24h, medication_7d, medication_m, sickness_status, id_subj, visit, measurement
        info_raw = self.info_normalize(subject_info)

        id_subj = self.id_subj_chunk[idx]
        file_path = self.rec_list_chunk[idx]

        return signal_s[:, :].astype(np.float32), signal_t[:, :].astype(np.float32), info.astype(np.float32), id_subj, dbp_true, sbp_true, info_raw, file_path

    def getitem_calibration(self, idx):
        couple_ = self.list_couple_train[idx]
        signal1_list = self.signal_chunk[couple_[0]].copy()  # cali data
        signal2_list = self.signal_chunk[couple_[1]].copy()  # pred data

        if self.split_type:
            # 以Rpeak分割数据
            if self.is_subseq_random:
                ix_cali = random.sample(self.ecg_r_peak_chunk[couple_[0]], 1)[0]
                cali_ix_len = self.sig_idx_len_chunk[couple_[0]][ix_cali[0]]
                ix_pred = random.sample(self.ecg_r_peak_chunk[couple_[1]], 1)[0]
                pred_ix_len = self.sig_idx_len_chunk[couple_[1]][ix_pred[0]]
            else:
                ix_cali = self.ecg_r_peak_chunk[couple_[0]][0]  # self.ecg_r_peak_chunk[couple_[0]][50]
                cali_ix_len = self.sig_idx_len_chunk[couple_[0]][ix_cali[0]]
                ix_pred = self.ecg_r_peak_chunk[couple_[1]][0]  # self.ecg_r_peak_chunk[couple_[1]][5]
                pred_ix_len = self.sig_idx_len_chunk[couple_[1]][ix_pred[0]]
        else:
            # 随机分割
            if self.is_subseq_random:
                cali_ix_len = random.sample(self.sig_idx_len_chunk[couple_[0]], 1)[0]  # 第几条，数据长度
                if cali_ix_len[1] > self.seq_len:
                    ix_cali = [cali_ix_len[0], random.randint(0, cali_ix_len[1] - self.seq_len - 1)]  # 第几条，数据起始
                else:
                    ix_cali = [cali_ix_len[0], 0]
                pred_ix_len = random.sample(self.sig_idx_len_chunk[couple_[1]], 1)[0]
                if pred_ix_len[1] > self.seq_len:
                    ix_pred = [pred_ix_len[0], random.randint(0, pred_ix_len[1] - self.seq_len - 1)]
                else:
                    ix_pred = [pred_ix_len[0], 0]
            else:
                cali_ix_len = self.sig_idx_len_chunk[couple_[0]][0]
                ix_cali = [cali_ix_len[0], 0]  # 从开头开始
                pred_ix_len = self.sig_idx_len_chunk[couple_[1]][0]
                ix_pred = [pred_ix_len[0], 0]  # 从开头开始

        if cali_ix_len[1] > ix_cali[1] + self.seq_len:
            signal1 = signal1_list[ix_cali[0]][:, ix_cali[1]:ix_cali[1] + self.seq_len]  # cali 随机r peak+len长度
        else:
            signal1 = signal1_list[ix_cali[0]][:, ix_cali[1]:]
            signal1 = np.pad(signal1, ((0, 0), (0, self.seq_len - signal1.shape[1])), 'constant', constant_values=0)
        if pred_ix_len[1] > ix_pred[1] + self.seq_len:
            signal2 = signal2_list[ix_pred[0]][:, ix_pred[1]:ix_pred[1] + self.seq_len]  # pred
        else:
            signal2 = signal2_list[ix_pred[0]][:, ix_pred[1]:]
            signal2 = np.pad(signal2, ((0, 0), (0, self.seq_len - signal2.shape[1])), 'constant', constant_values=0)

        ch = {'ECG': 0, 'PPG_IR': 1, 'PPG_Y': 2, 'PPG_B': 3, 'Force': 4, 'ECG_Clean': 5,
              'ECG_Phase_Completion_Atrial': 6, 'ECG_Phase_Completion_Ventricular': 7,
              }
        force1 = signal1[None, 4]
        force2 = signal2[None, 4]

        # signal1_ = np.concatenate((signal1_[None, 1], signal1_[None, 2], signal1_[None, 3], signal1_[None, 5], signal1_[None, 4], signal1_[None, 7]), axis=0)
        # signal2_ = np.concatenate((signal2_[None, 1], signal2_[None, 2], signal2_[None, 3], signal2_[None, 5], signal2_[None, 4], signal2_[None, 7]), axis=0)
        if self.force_flag:
            signal1_ = np.concatenate((signal1[None, 1], signal1[None, 2], signal1[None, 3],  # ppg ac
                                       signal1[None, 5], signal1[None, 4],
                                       signal1[None, 6], signal1[None, 7]), axis=0)
            signal2_ = np.concatenate((signal2[None, 1], signal2[None, 2], signal2[None, 3],
                                       signal2[None, 5], signal2[None, 4],
                                       signal2[None, 6], signal2[None, 7]), axis=0)
            if self.info_chunk[couple_[0]]['watch_type'] == 2 or (self.info_chunk[couple_[0]]['watch_type'] != -1 and (
                    np.max(signal1_[4, :]) - np.min(signal1_[4, :]) == 0 or np.isnan(signal1_[4, :]).any())):
                signal1_[4, :] = np.random.rand(signal1_.shape[1]) + 70
            if self.info_chunk[couple_[0]]['watch_type'] == 2 or (self.info_chunk[couple_[1]]['watch_type'] != -1 and (
                    np.max(signal2_[4, :]) - np.min(signal2_[4, :]) == 0 or np.isnan(signal2_[4, :]).any())):
                signal2_[4, :] = np.random.rand(signal2_.shape[1]) + 70
        else:
            signal1_ = np.concatenate((signal1[None, 1], signal1[None, 2], signal1[None, 3],  # ppg ac
                                       signal1[None, 5], signal1[None, 6], signal1[None, 7]), axis=0)
            signal2_ = np.concatenate((signal2[None, 1], signal2[None, 2], signal2[None, 3],
                                       signal2[None, 5], signal2[None, 6], signal2[None, 7]), axis=0)
            # signal1_ = np.concatenate((signal1_[None, 1], signal1_[None, 2], signal1_[None, 3]), axis=0)
            # signal2_ = np.concatenate((signal2_[None, 1], signal2_[None, 2], signal2_[None, 3]), axis=0)
        norm_ch = [0, 1, 2, 3]

        sig_mean1 = signal1_[norm_ch].mean(1, keepdims=True)
        sig_std1 = np.clip(signal1_[norm_ch].std(1, keepdims=True), 1e-9, np.max(np.abs(signal1_[norm_ch])))
        signal1_[norm_ch] = (signal1_[norm_ch] - sig_mean1) / sig_std1
        # signal1_[5] = (signal1_[5] - 0.5) / 1     # R peak
        # if self.force_flag:
        #     signal1_[4] = (signal1_[4] - 70.0) / 10.0

        sig_mean2 = signal2_[norm_ch].mean(1, keepdims=True)
        sig_std2 = np.clip(signal2_[norm_ch].std(1, keepdims=True), 1e-9, np.max(np.abs(signal2_[norm_ch])))
        signal2_[norm_ch] = (signal2_[norm_ch] - sig_mean2) / sig_std2
        # signal2_[5] = (signal2_[5] - 0.5) / 1
        # if self.force_flag:
        #     signal2_[4] = (signal2_[4] - 70.0) / 10.0

        signal = np.stack((signal1_, signal2_), axis=0)
        # signal = np.concatenate((signal1_, signal2_), axis=0)

        # info normalization     age, gender, weight, height, watch_type, force cali_sbp/pred_sbp
        info_ = np.zeros((2, 10))
        subject_info_cali = self.info_chunk[couple_[0]]
        subject_info_pred = self.info_chunk[couple_[1]]
        info_[:, 0] = subject_info_cali['age']
        info_[:, 1] = subject_info_cali['gender']
        info_[:, 2] = subject_info_cali['height']
        info_[:, 3] = subject_info_cali['weight']
        info_[0, 6] = np.mean(force1)
        info_[1, 6] = np.mean(force2)
        # info_[:, 6] = subject_info_cali['watch_type']
        info_[0, 7] = subject_info_cali['medication_status_24hours']
        info_[1, 7] = subject_info_pred['medication_status_24hours']
        info_[0, 8] = subject_info_cali['medication_status_7day']
        info_[1, 8] = subject_info_pred['medication_status_7day']
        # info_[0, 9] = int((subject_info_cali['datetime'].hour - 12) // 4)
        # info_[1, 9] = int((subject_info_pred['datetime'].hour - 12) // 4)
        t1 = subject_info_cali['datetime']
        t2 = subject_info_pred['datetime']
        delta_day = abs((t2-t1).days)
        info_[0, 9] = delta_day
        info_[1, 9] = delta_day

        if couple_[0] == couple_[1]:
            info_[0, 4] = np.array(0.0)
            info_[1, 4] = np.array(0.0)
            info_[0, 5] = np.array(0.0)
            info_[1, 5] = np.array(0.0)
        else:
            info_[0, 4] = np.array(np.clip(self.dbp_true_chunk[couple_[0]], 43.0, 220.0))  # cali
            info_[1, 4] = np.array(np.clip(self.dbp_true_chunk[couple_[0]], 43.0, 220.0))  # pred
            info_[0, 5] = np.array(np.clip(self.sbp_true_chunk[couple_[0]], 43.0, 220.0))
            info_[1, 5] = np.array(np.clip(self.sbp_true_chunk[couple_[0]], 43.0, 220.0))

        info = (info_ - np.array([40.0, 0.0, 160.0, 60.0, 80.0, 120.0, 70.0, 0.0, 0.0])) / np.array(
            [20.0, 1.0, 25.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0])

        # info_mean = np.concatenate((self.info_mean[:4], np.array([73.40, 125.36])))
        # info_std = np.concatenate((self.info_std[:4], np.array([10.66, 20.14])))
        # info = (info_ - info_mean) / info_std

        dbp_true = np.clip(self.dbp_true_chunk[couple_[1]], 43.0, 220.0)  # pred
        sbp_true = np.clip(self.sbp_true_chunk[couple_[1]], 43.0, 220.0)

        # age, gender, height, weight, wrist_circumference, medication_24h, medication_7d, medication_m, sickness_status, id_subj, visit, measurement
        info_raw = self.info_normalize(self.info_chunk[couple_[1]])
        id_subj = self.id_subj_chunk[couple_[1]]
        file_path = self.rec_list_chunk[couple_[1]]

        assert len(signal) > 0

        return signal, info, id_subj, dbp_true, sbp_true, info_raw, file_path

    def __len__(self):
        if self.cali_flag == 'free':
            chunk_len = len(self.sbp_true_chunk)
        else:
            chunk_len = len(self.list_couple_train)
        return chunk_len

    def __getitem__(self, idx):
        if self.cali_flag == 'free':
            signal_s, signal_t, info, id_subj, dbp_true, sbp_true, info_raw, file_path = self.getitem_free(idx)
        else:
            signal_s, signal_t, info, id_subj, dbp_true, sbp_true, info_raw, file_path = self.getitem_calibration(idx)

        return signal_s.astype(np.float32), signal_t.astype(np.float32), info.astype(np.float32), id_subj, dbp_true, sbp_true, info_raw, file_path



def get_data_loader(args):
    """
    :param args: config list
    :param construct_data: callback function for construct special format for model
    :param class_dataset: self-dataset inherited from torch.utils.data.Dataset
    :return: iter_train, iter_eval, iter_test
    """

    train_loader = None
    eval_loader = None
    test_loader = None
    if args.mode == 'pretrain' or args.mode == 'finetune':
        seq_len = args.seq_len
        train_path_list = [args.train_dataset_path, args.train_dataset_path_lab, args.train_dataset_path_lange,
                           args.train_dataset_path_med1000]
        # train_path_list = [args.train_dataset_path_lab]
        # train_path_list = [args.train_dataset_path, args.train_dataset_path_lange]
        # train_path_list = [args.val_dataset_path1, args.val_dataset_path2, args.val_dataset_path3]

        train_data = BpDataset('train', args.r_peak_split, args.gold_relisten, train_path_list,
                               idx_chunk_init=0, is_subseq_random=True, is_fix_len=True, seq_len=seq_len,
                               cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                               norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = torch.utils.data.RandomSampler(train_data)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=sampler_train,
                                  num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
                                  collate_fn=None)

        seq_len_eval = args.seq_len
        # eval_path_list = [args.val_dataset_path1]
        eval_path_list = [args.val_dataset_path1, args.val_dataset_path2, args.val_dataset_path3]

        eval_data = BpDataset('train', args.r_peak_split, args.gold_relisten, eval_path_list,
                              idx_chunk_init=0, is_subseq_random=True, is_fix_len=True, seq_len=seq_len_eval,
                              cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                              norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        eval_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_mem, collate_fn=None)

    return train_loader, eval_loader, test_loader


def get_finetune_data_loader(args):
    """
    :param args: config list
    :param construct_data: callback function for construct special format for model
    :param class_dataset: self-dataset inherited from torch.utils.data.Dataset
    :return: iter_train, iter_eval, iter_test
    """

    train_loader = None
    eval_loader = None
    test_loader = None
    if (args.mode == 'pretrain' or args.mode == 'finetune' or args.mode == 'linprobe') and not args.eval:
        seq_len = args.seq_len
        train_path_list = [args.train_dataset_path, args.train_dataset_path_lab, args.train_dataset_path_lange,
                           args.train_dataset_path_med1000]
        # train_path_list = [args.train_dataset_path]
        # train_path_list = [args.val_dataset_path1, args.val_dataset_path2, args.val_dataset_path3]

        train_data = BpDataset('train', args.r_peak_split, args.gold_relisten, train_path_list,
                               idx_chunk_init=0, is_subseq_random=True, is_fix_len=True, seq_len=seq_len,
                               cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                               norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = torch.utils.data.RandomSampler(train_data)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=sampler_train,
                                  num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
                                  collate_fn=None)

        seq_len_eval = args.seq_len
        eval_path_list = [args.val_dataset_path1, args.val_dataset_path2, args.val_dataset_path3]

        eval_data = BpDataset('evaluate', args.r_peak_split, args.gold_relisten, eval_path_list,
                              idx_chunk_init=0, is_subseq_random=True, is_fix_len=True, seq_len=seq_len_eval,
                              cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                              norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                eval_data, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_val = torch.utils.data.SequentialSampler(eval_data)
        eval_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, sampler=sampler_val,
                                 num_workers=args.num_workers, pin_memory=args.pin_mem, collate_fn=None)

        seq_len_test = args.seq_len
        # test_path_list = [args.test_dataset_path_0329]
        test_path_list = [args.test_dataset_path_0329, args.test_dataset_path_0404,
                          args.test_dataset_path_round0, args.test_dataset_path_round1, args.test_dataset_path_round2,
                          args.test_dataset_path_round3, args.test_dataset_path_round4, args.test_dataset_path_round5,
                          args.test_dataset_path_round6, args.test_dataset_path_round7, args.test_dataset_path_round8]

        test_data = BpDataset('predict', args.r_peak_split, args.gold_relisten, test_path_list,
                              idx_chunk_init=0, is_subseq_random=False, is_fix_len=True, seq_len=seq_len_test,
                              cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                              norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        if args.dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(
                test_data, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_test = torch.utils.data.SequentialSampler(test_data)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=None)
    if args.eval:
        seq_len_test = args.seq_len
        # test_path_list = [args.val_dataset_path1, args.val_dataset_path2, args.val_dataset_path3]
        # test_path_list = [args.test_dataset_path_0329]
        # test_path_list = [args.test_dataset_path_PDU85]
        # test_path_list = [args.test_dataset_path_Med50]
        # test_path_list = [args.test_dataset_path_PDU35]
        # test_path_list = [args.test_dataset_path_Med1000]
        test_path_list = [args.test_dataset_path_0329, args.test_dataset_path_0404,
                          args.test_dataset_path_round0, args.test_dataset_path_round1, args.test_dataset_path_round2,
                          args.test_dataset_path_round3, args.test_dataset_path_round4, args.test_dataset_path_round5,
                          args.test_dataset_path_round6, args.test_dataset_path_round7, args.test_dataset_path_round8]

        test_data = BpDataset('predict', args.r_peak_split, args.gold_relisten, test_path_list,
                              idx_chunk_init=0, is_subseq_random=False, is_fix_len=True, seq_len=seq_len_test,
                              cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                              norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=None)

    return train_loader, eval_loader, test_loader


def get_test_data_loader(args, test_path_list, cali_type):
    test_data = BpDataset('predict', args.r_peak_split, args.gold_relisten, test_path_list, idx_chunk_init=0,
                          is_subseq_random=False, is_fix_len=True, seq_len=args.seq_len,
                          cali_flag=args.CALIBRATION, force_flag=args.FORCE_FLAG,
                          norm_tag_s=args.norm_tag_s, norm_tag_t=args.norm_tag_t, cali_type=cali_type)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=None)

    return test_loader
