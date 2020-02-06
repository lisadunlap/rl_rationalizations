"""
Laurens Weitkamp

run the game once, creating grad-cam files in the process

python3 main.py --env-name ENV_NAME --gradcam_layer GCAM_LAYER

This command should run an episode of ENV_NAME using the pretrained models
and collect Grad-CAM outputs for each state/action at layer GCAM_LAYER
(which would default to features.elu4).
"""
import torch
import gym
import pygame
import sys
import time
import matplotlib
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import argparse

from play import *
from Grad_CAM import Grad_CAM
from model import ActorCritic
from rollout import *
from test import test
from envs import *

def get_cam(img, mask):

    '''
    Place MASK heatmap on IMG
    '''
    h, w, _ = img.shape
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    alpha = .5
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    return cam


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--env', default='Pong',
                        help='Name of environment.', choices=['Pong', 'Seaquest', 'BeamRider'])
    parser.add_argument('--half', action='store_true',
                        help='Use half trained model')

    args = parser.parse_args()


    env_name = args.env+'-v0'
    #create save directory for gradcams
    if not os.path.exists(env_name+'/'):
        os.makedirs(env_name+'/')

    print("set up dir variables and environment...")
    load_dir = "pretrained/A3C_{}_Full.pth".format(args.env)
    env = create_atari_env(env_name)
    env.seed(1)

    print("initialize agent and try to load saved weights...")
    model = ActorCritic(num_inputs=env.observation_space.shape[0], action_space=env.action_space)
    _ = model.load_state_dict(torch.load((load_dir)))
    torch.manual_seed(1)
    print('testing...')
    
    # run simulation and generate gradcams for every action
    history = test(env_name, model)