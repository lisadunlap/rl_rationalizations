import time
from collections import deque

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torch.autograd import Variable
import sys

from envs import create_atari_env
from Grad_CAM import Grad_CAM
from model import ActorCritic

rescale = lambda image: np.uint8(((image-np.min(image))/np.max((image-np.min(image))))*255)

def get_cam(img, mask):

    '''
    Place MASK heatmap on IMG
    '''
    h, w, _ = img.shape
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    alpha = .4
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    return np.uint8(cam*255)


def test(env_name, shared_model, max_episode_length=1000):
    #generate gradcam model
    gcam = Grad_CAM(model=shared_model)

    history = {'ins': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
    torch.manual_seed(1)

    env = create_atari_env(env_name)
    env.seed(1)
    action_labels = env.unwrapped.get_action_meanings()

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state, frame = env.reset()
    state, frame  = torch.from_numpy(state), torch.from_numpy(frame)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0)), hx, cx)
        probs, ids = gcam.forward(Variable(state.unsqueeze(0), requires_grad=True), (hx, cx))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        (state, frame), reward, done, _ = env.step(action[0, 0])

        # compute gradcams
        gcam.backward(idx=ids[0])
        regions = gcam.generate(target_layer='features.elu4')
        cam = get_cam(frame, regions)
        filename = "{0}/{1}-{2}.png".format(env_name, episode_length, action_labels[ids[0]])
        cv2.imwrite(filename, cam)

        done = done or episode_length >= max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                1, 1 / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            return history

        state = torch.from_numpy(state)