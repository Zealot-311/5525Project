# Model Descriptions

## The following document details relevant hyperparameters and other important facts for each of the stored .pickle files. For all such files, use `joblib.load()` to load the model.

### `21_softmaxoutput.pickle`

This model used the Actor Critic framework, with full RGB inputs, a 128 node hidden layer (ReLU) and 12 outputs (softmax). This is the result of the 21st episode, at around 2 hours of training.

These 12 outputs were taken as the probability to press each button independently (this is wrong, in hindsight), thus we then run a bernoulli trial for each of the twelve possible actions and press the indicated button if the corresponding trial is positive.

One may see that this is wrong from the fact that, while softmax can be interpreted as a probability, you will most definitely always get one output that is much larger than the others, since the sum of all the probabilities across all outputs is 1 (and thus cannot be treated for independent bernoulli trials).

We should thus consider other options that would better represent independent probabilities (that together may combine to do more interesting things).

Additionally, the punishments and rewards went as follows:
```python
# Parameters for reward function
score_reward = 1
damage_punishment_constant = 0.1
done_damage_reward = 10
survival_reward = 1

# Where
reward += score_reward * score_increase
reward += damage_punishment_constant * (current_health - initial_health)
reward += done_damage_reward * (initial_enemy_health - current_enemy_health)
reward += survival_reward
```
`score_increase` is just the default value that `reward` gets.

Initially I accidentally forgot to update the health variables, but it turns out it's actually not worse than continually updating them so 🤷.


### `27_sigmoid_update.pickle`
Again, Actor Critic framework but this time with grayscale inputs. This model both trains faster and runs faster when testing playback, without any noticeable impact on effectiveness. The output now passes through the sigmoid (logistic) function, to simulate having an independent probability for each output. This means outputs are less consistent (i.e. not having the same button pressed over and over for the entire duration of the runtime) but at the same time they are more complex. This model at episode 27 has learned several combinations of buttons that perform higher damage attacks, along with learning to avoid ranged attacks from the enemy. Given the nature of how this game is trained, it performs pretty well against the first opponent and can regularly beat it, but it suffers a bit against the second opponent (some attacks _look_ different, I think this is the root cause of the issue). Beating the second enemy consistently would mark a milestone in the model's training since the best we've seen the random inputs do is lose twice in a row to the second opponent.

```py
# Parameters for reward function
score_reward = 1 
damage_punishment = 0.1 
done_damage_reward = 10
survival_reward = 0
win_reward = 100_000 
lose_punishment = -100_000

# Where
reward += score_reward * score_increase 
reward += damage_punishment * (current_health - previous_health)
reward += done_damage_reward * (prev_enemy_health - current_enemy_health)
reward += current_health * survival_reward 
reward += lose_punishment * (current_enemy_wins - prev_enemy_wins)
reward += win_reward * (prev_wins - current_wins)
```
This model was implemented with updates to the respective variables.

### `31_sigmoid_noupdate_defensive.pickle`
This model was trained with almost the same hyperparameters and reward function to above, except that it was trained without updates to the respective variables.

```py
# Parameters for reward function
score_reward = 1 
damage_punishment = 0.1 
done_damage_reward = 1
survival_reward = 0
win_reward = 100_000 
lose_punishment = -100_000

# Where
reward += score_reward * score_increase 
reward += damage_punishment * (current_health - initial_health)
reward += done_damage_reward * (initial_enemy_health - enemy_health)
reward += lose_punishment * (current_enemy_wins)
reward += win_reward * (0 - current_wins)
```

The astute among us might notice that I made a severe mistake here: The `win_reward` is negative whenever the model wins a match. Whoops! I didn't notice earlier because it was masked by variable names and I wasn't thinking correctly. Thus, it would make sense that this model essentially just tries to not lose for as long as possible. However, on the bright side, this means the model learned to block and avoid damage as much as possible! Another thing that contributed to this was that the `done_damage_reward` was accidentally set to 1, meaning the model did not seek to do damage as often. This model was stopped at episode 31.

### `54_sigmoid_noupdate_attacking.pickle`
This model was trained with almost the same hyperparameters and reward function to above, except that the survival reward was increased to 1 and the `done_damage_reward` was fixed.

```py
# Parameters for reward function
score_reward = 1 
damage_punishment = 0.1 
done_damage_reward = 10
survival_reward = 1
win_reward = 100_000 
lose_punishment = -100_000

# Where
reward += score_reward * score_increase 
reward += damage_punishment * (current_health - initial_health)
reward += done_damage_reward * (initial_enemy_health - enemy_health)
reward += lose_punishment * (current_enemy_wins)
reward += win_reward * (0 - current_wins)
```
Again, I still hadn't figured out the issue with the win reward, so you'll pardon me for a second. The one difference maker here was the `done_damage_reward`. No longer does the model try and defend itself infinitely, since apparently a reward multiplier of 10 is enough to overcome the problem with the reward for winning being negative, since this model is able to consistently get past the first enemy, and sometimes even win one of the required two wins against the second. This model had similar issues to [this model](#`27_sigmoid_update.pickle`)

### Note on update vs noupdate

So far, between `27_sigmoid_update.pickle` and `54_sigmoid_noupdate_attacking.pickle` it's hard to tell a difference. I will say, that on average these models do not get past the second enemy, which might contribute to the non-difference between the two. I think that with enough training, the first model will perform better because it won't have the lagging match loss punishment hanging behind it. Also consider that they are indistinguishable when the second has almost twice as many episodes. This could be because the update learns faster, or it could be because there is a plateau when using this algorithm. The problem here was I was trying to train several models at once, but the downside to this is that their processes were killed at different points because of memory. The two numbers at the start of the filename represent the _last_ episode completed and stored by the script. Only training them independently for more time would be able to tell us what the problem was. However, I want to fix the `win_reward` being negative, so we will stop considering these specific models in favor of the corrected model.

The new reward value for the wins will be 
```py
reward += win_reward * (current_wins - prev_wins)
prev_wins = current_wins
```


### `66_update_surv_1.pickle` vs `66_update_surv_0.pickle`: Testing survival reward

Both models have grayscale input, sigmoidal output.

Hyperparameters for the first are
```py
# Parameters for reward function
score_reward = 1 # Multiplies the base reward value (given by the change in score)
damage_punishment = 0.1 # Multiplies the damage taken since last frame
done_damage_reward = 10 # Multiplies the damage done since last frame
survival_reward = 1 # Increase reward by this much times current health
win_reward = 100_000 # Increase reward by this much if the model won a match in this frame
lose_punishment = -100_000 # Decrease reward by this much if the model lost a match in this frame
```
Notes:
* Just jumps over and over again. Avoids damage, tries to increase time survived (probably).
* Did not beat first opponent ever in 10-15 minutes left testing.


Hyperparameters for the second are
```py
 # Parameters for reward function
score_reward = 1 # Multiplies the base reward value (given by the change in score)
damage_punishment = 0.1 # Multiplies the damage taken since last frame
done_damage_reward = 10 # Multiplies the damage done since last frame
survival_reward = 0 # Increase reward by this much times current health
win_reward = 100_000 # Increase reward by this much if the model won a match in this frame
lose_punishment = -100_000 # Decrease reward by this much if the model lost a match in this frame
```
Notes: 
* Does do some damage. However, does not defend well enough so it just dies sometimes.
* Did beat the first opponent at least once. 
* The astute may realize this is the same model as `27_sigmoid_update.pickle` except for the fact that the `win_reward` is calculated correctly. Is the other one better because of this reason? Is the other one better simply by luck? Is the other one just not better but it seemed as though it was? Not sure.

Computation for current reward was calculated similarly to before.


