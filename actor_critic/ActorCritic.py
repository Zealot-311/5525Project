# Code and model structure adapted from https://keras.io/examples/rl/actor_critic_cartpole/

# import gym
import retro
# import pickle
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def to_grayscale(img_mat):
    return img_mat @ np.array([0.2989, 0.5870, 0.1140])

def main():

    ################ SETUP ################
    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = 100000
    env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")  # Create the environment
    env.seed(seed)
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0



    ################ ACTOR-CRITIC NETWORK ################
    num_inputs = np.prod(env.reset().shape) // 3 # Calculate size of grayscale image
    num_actions = env.action_space.shape[0]
    # possible_actions = env.action_space
    num_hidden = 128

    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="sigmoid")(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=[action, critic])

    # Parameters for reward function
    score_reward = 1 # Multiplies the base reward value (given by the change in score)
    damage_punishment = 0.1 # Multiplies the damage taken since last frame
    done_damage_reward = 10 # Multiplies the damage done since last frame
    survival_reward = 0 # Increase reward by this much times current health
    win_reward = 100_000 # Increase reward by this much if the model won a match in this frame
    lose_punishment = -100_000 # Decrease reward by this much if the model lost a match in this frame


    ################# TRAIN ###################
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    # model.compile()


    while True:  # Run indefinitely
        env.reset()

        # Initialize some variables
        prev_health = env.step(env.action_space.sample())[3]['health']
        prev_enemy_health = env.step(env.action_space.sample())[3]['health']
        prev_wins = 0
        prev_enemy_wins = 0

        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):

                # env.render() # Render current frame
                state = to_grayscale(state)
                state = tf.convert_to_tensor(state)
                state = tf.reshape(state, (num_inputs,))
                state = tf.expand_dims(state, 0)
                
                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])
                action_probs = tf.reshape(action_probs, (num_actions,))

                # Sample action from action probability distribution
                action = np.asarray([1 if (np.random.uniform(0.0, 1.0) <= prob) else 0 for prob in action_probs])

                # The probability of taking this action is the product of the binomial distributions
                this_action_prob = np.prod([1 - prob if action[i] == 0 else prob for i, prob in enumerate(action_probs)])
                action_probs_history.append(tf.math.log(this_action_prob))

                # Apply the sampled action in our environment
                state, reward, done, _ = env.step(action)

                # Make the reward take into account other things too!
                reward *= score_reward # This one always first or else >:(
                
                reward += damage_punishment * (_['health'] - prev_health)
                prev_health = _['health']
                reward += done_damage_reward * (prev_enemy_health - _['enemy_health'])
                prev_enemy_health = _['enemy_health']
                reward += _['health'] * survival_reward # Survival reward only happens if the health is not zero (also if your health is higher you get better reward?)
                reward += lose_punishment * (_['enemy_matches_won'] - prev_enemy_wins)
                prev_enemy_wins = _['enemy_matches_won']
                reward += win_reward * (_['matches_won'] - prev_wins)
                prev_wins = _['matches_won']
                
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

                
            
            print(f"Finished episode")
            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()

            # Calculating loss values to update our network
            actor_losses = []
            critic_losses = []
            for ret, crit, act in zip(returns, critic_value_history, action_probs_history):
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - crit
                actor_losses.append(-act * diff)

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(huber_loss(tf.expand_dims(crit, 0), tf.expand_dims(ret, 0)))
            
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

            # Log details
            episode_count += 1
            # if episode_count % 10 == 0:
            print(f"running reward: {running_reward:.2f} at episode {episode_count}")
            print(f'episode loss of {episode_reward:e}')

            joblib.dump(model, f"actor_critic_models/ac_{episode_count}_{int(round(episode_reward)):e}.pickle")

        # if running_reward > 195:  # Condition to consider the task solved
        #     print("Solved at episode {}!".format(episode_count))
        #     break






if __name__ == '__main__':
    main()

