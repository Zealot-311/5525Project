import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F
import WorkerUtils as wu
import retro
import gym
import traceback
import logging

logging.basicConfig(filename='logs/pytorch_stats.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# The worker class for running agent vs Computer training, aka story mode training
class Worker(mp.Process):

    def __init__(self, env_id, roms_path, epoch_size, model, optim, criterion, rewardQueue, frameRatio, framesPerStep):
        super(Worker, self).__init__()
        self.env_id = env_id
        self.roms_path = roms_path
        self.epoch_size = epoch_size
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.rewardQueue = rewardQueue
        self.frameRatio = frameRatio
        self.framesPerStep = framesPerStep

    def run(self):
        try:
            logger.info("Starting Worker")
            env = retro.make(self.env_id, record='rec/')
            self.env = gym.wrappers.ResizeObservation(env, (182,360))
            initial_obs = self.env.reset()
            while True:
                self.model.eval()

                observations, histories, frames = self.generate_playthrough(initial_obs)

                self.model.train()

                dataset = wu.compileHistories(observations, histories)

                wu.train(self.model, self.optim, self.criterion, dataset)

        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
    
    def generate_playthrough(self, initial_obs):
        observations = [[]]
        histories = [{"moveAction": [], "attackAction": [], "reward": []}]
        epoch_reward = 0
        total_round = 0
        done = False
        frames = []

        for i in range(self.epoch_size):
            for k in range(self.framesPerStep):
                frames.append(initial_obs)

            while not done:
                x = wu.prepro(frames)

                observations[total_round].append(x)
                moveOut, attackOut = self.model(Variable(x))

                moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))

                attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))

                histories[total_round]["moveAction"].append(torch.FloatTensor(1).fill_(moveAction))

                histories[total_round]["attackAction"].append(torch.FloatTensor(1).fill_(attackAction))

                frames = []
                action = wu.map_action(moveAction, attackAction)
                action_reward = 0
                for j in range(self.framesPerStep):
                    if(j < self.framesPerStep-1):
                        obs, rew, done, info = self.env.step(action)
                    else:
                        obs, rew, done, info = self.env.step(wu.map_action(moveAction))
                    
                    # changes to reward
                    rew = info['score']*info['matches_won']
                    
                    frames.append(obs)
                    action_reward += rew
                    epoch_reward += rew
        
                    if done:
                        break
                histories[total_round]["reward"].append(torch.FloatTensor(1).fill_(action_reward))

            total_round += 1
            histories.append({"moveAction": [], "attackAction": [], "reward": []})
            self.rewardQueue.put({"reward": epoch_reward})

            initial_obs = self.env.reset()

        return observations, histories, frames