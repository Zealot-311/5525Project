import gym
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import retro
import WorkerUtils as wu
from torch.autograd import Variable
from Model import Model
import torch
import torch.nn.functional as F
import imageio
import numpy as np




def prepro(frame, isGrey):
    # frame = frame[32:214, 12:372]  # crop
    frame = frame[::3, ::3]
    if isGrey:
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
    return frame


def setupModel(episode, framesPerStep, loadPath):
    model = Model(framesPerStep, 4, 6)
    if episode > 0:  # For loading a saved model
        model.load_state_dict(torch.load(loadPath + "models/" + str(episode), map_location=lambda storage, loc: storage))
    return model

isGrey = True
framesPerStep = 3
loadPath = 'saves/'
savePath = 'vid/'
env_id = 'StreetFighterIISpecialChampionEdition-Genesis'
# set to model to load
episode = 400
show_viz = True

model = setupModel(episode, framesPerStep, loadPath)

env = retro.make(env_id)
env = gym.wrappers.ResizeObservation(env, (182,360))

if show_viz:
    fig = plt.figure()
    plt.ion()
    obs = env.reset()
    im: AxesImage = plt.imshow(prepro(obs, isGrey), cmap="gray" if isGrey else None)
    plt.axis("off")
    plt.show()
done = False

frames = [env.reset(), env.reset(), env.reset()]
images = [env.render(mode='rgb_array')]

while not done:
    x = wu.prepro(frames)
    moveOut, attackOut = model(Variable(x))
    moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))
    attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))
    action = wu.map_action(moveAction, attackAction)
    
    frames = []
    for j in range(framesPerStep):
        if(j < framesPerStep-1):
            obs, rew, done, info = env.step(action)
        else:
            obs, rew, done, info = env.step(wu.map_action(moveAction))
        frames.append(obs)
        images.append(env.render(mode='rgb_array'))
        if show_viz:
            im.set_data(prepro(obs, isGrey))
            plt.pause(0.00001)
    
imageio.mimsave(env_id + '_sfagent.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=1000*1/29)