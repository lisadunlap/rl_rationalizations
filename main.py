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

from play import *
from Grad_CAM import Grad_CAM
from model import ActorCritic
from rollout import *

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
    env_name = 'Pong-v0'
    save_dir = 'figures/'

    print("set up dir variables and environment...")
    load_dir = 'pretrained/A3C_Pong_Half.pth'
    env = gym.make(env_name);
    env.seed(1)

    print("initialize agent and try to load saved weights...")
    model = ActorCritic(num_inputs=1, action_space=env.action_space)
    _ = model.load_state_dict(torch.load((load_dir)));
    torch.manual_seed(1)
    torch.manual_seed(1)

    history = rollout(model, env, max_ep_len=3e3)
    imgs = []
    actions1 = []
    actions2 = []
    actions3 = []
    cams = []
    cams2 = []

    for frame_ix in range(200, 210):
        img = history['ins'][frame_ix]

        # get input
        tens_state = torch.Tensor(prepro(img))
        state = Variable(tens_state.unsqueeze(0), requires_grad=True)
        hx = Variable(torch.Tensor(history['hx'][frame_ix - 1]).view(1, -1))
        cx = Variable(torch.Tensor(history['cx'][frame_ix - 1]).view(1, -1))

        # Grad-CAM
        gcam = Grad_CAM(model=model)
        action_labels = env.unwrapped.get_action_meanings()
        probs, ids = gcam.forward(state, (hx, cx))
        actions1 += [action_labels[ids[0]]]
        actions2 += [action_labels[ids[0]]]
        actions3 += [action_labels[ids[1]]]
        print('probs: {0}   ids: {1}  {2}'.format(probs, action_labels[ids[0]], action_labels[ids[1]]))

        # backprop and generate gcam
        gcam.backward(idx=ids[0])
        regions = gcam.generate(target_layer='features.elu4')

        # plot images
        cam = get_cam(img[35:195], regions)
        imgs += [img[35:195]]
        cams += [cam]

        gcam.backward(idx=ids[1])
        regions = gcam.generate(target_layer='features.elu4')
        cam = get_cam(img[35:195], regions)
        cams2 += [cam]

        fig = plt.figure(figsize=(40, 15))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 10),
                         axes_pad=0.3,  # pad between axes in inch.
                         )
        actions = actions1+actions2+actions3

        j = 0
        for ax, im in zip(grid, imgs+cams+cams2):
            ax.axis('off')
            ax.set_title(actions[j])
            ax.imshow(im)
            j += 1

        fig.savefig(save_dir+"/pong.png")