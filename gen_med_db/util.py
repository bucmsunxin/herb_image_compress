# -*- coding: utf-8 -*-
import cv2
import pickle
import torch
import numpy as np
import math
import json
import random
import os
import os.path as osp
import pdb
import copy
import scipy.io as sio
import pdb

def create_dir(tmp_dir):
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)

def load_file(data_file):
    ptr = open(data_file, 'rb')
    data = pickle.load(ptr)
    return data

def save_file(data, data_file):
    ptr = open(data_file, 'wb')
    pickle.dump(data, ptr)
    ptr.close()
