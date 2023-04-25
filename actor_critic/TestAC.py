import retro
# import pickle
import joblib
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

def to_grayscale(img_mat):
    return img_mat @ np.array([0.2989, 0.5870, 0.1140])

def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

    obs = env.reset()
    model = None
    
    # Change this to the model you want to load
    fname = "saved_models/66_update_surv_0.pickle"

    model = joblib.load(fname)

    num_inputs = np.prod(env.reset().shape) // 3
    num_actions = env.action_space.shape[0]

    if not (model is None):
        state = env.reset()
        # reward = 0
        while True:
            state = to_grayscale(state)
            state = tf.convert_to_tensor(state)
            state = tf.reshape(state, (num_inputs,))
            state = tf.expand_dims(state, 0)

            action_probs, _ = model(state)
            action_probs = tf.reshape(action_probs, (num_actions,))

            # Sample action from action probability distribution
            action = np.asarray([1 if (np.random.uniform(0.0, 1.0) <= prob)  else 0 for prob in action_probs])

            # print(action)

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)

            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()