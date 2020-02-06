# Explainable Reinforcement Learning: Visual Policy Rationalizations Using Grad-CAM
This reposity is an adaptation of https://github.com/lweitkamp/rl_rationalizations, who modifed GradCAM for A3C and trained the A3C models.
The Grad-CAM implementation used is based on https://github.com/kazuto1011/grad-cam-pytorch,
the A3C implementation is based on https://github.com/ikostrikov/pytorch-a3c.

## Examples
Some example analysis of the original author can be found [here](https://lweitkamp.github.io/thesis2018.html).

## How to use the code
This command plays a portion of an episode of a game (idk if that's the correct terminology but you know what I mean) using 
a pretrained model and saves the gradcams for each state/action at layer GCAM_LAYER (which would default to features.elu4). You can visualize these results [here](Visualize.ipynb).

You can choose to either used a fully trained model or a half trained model (seaquest only has a fully trained model)

```
python3 main.py --env [environment name] --half [use half trained]
```
Here's an example for pong:
```
python3 main.py --env Pong
```


## trained models
The models can be found in the pretrained folder. Each model has both a 'full' and a 'half' version (see paper).

|                    | Full Agent Mean | Full Agent Variance | Half Agent Mean | Half Agent Variance | DeepMind |
|--------------------|:---------------:|---------------------|-----------------|---------------------|----------|
| *Pong*      |           21.00 | 0.00                | 14.99           | 0.09                | 10.7         |
| *BeamRider* |         4659.04 | 1932.58             | 1597.40         | 1202.00             | 24622.2         |
| *Seaquest*  |         1749.00 | 11.44               | N/A             | N/A                 | 1326.1         |


Note that these models are based on amount of frames whereas DeepMind is based on 4 day training on 16 CPU cores, which
makes comparing them hard.
