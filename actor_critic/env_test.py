import retro
import time

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

obs = env.reset()
prev_health = env.step(env.action_space.sample())[3]['health']
obs = env.reset()
while True:
    action = env.action_space.sample()

    obs, rew, done, info = env.step(action)
    rew = rew + (info['health'] - prev_health)

    print(action)

    env.render()
    if done:
        obs = env.reset()

    time.sleep(1/60)