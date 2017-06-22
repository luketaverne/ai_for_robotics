import pyglet
import tensorflow as tf
import time
import numpy as np
import gym
import pylab as pl
import os
import collections

from NNModel import *
import Support as sup

# Set up the simulation environment
env = gym.make('CartPole-v0')
env.reset()
# env.render()
env.close()

#### HYPERPARAMTERS ####
# Training
batch_size = 16
learning_rate = 0.03
gamma = 0.99  # discount factor for reward
max_num_episodes = 10000
# TODO: add additional abort criterion
max_deflection = 15 # degrees
max_displacement = 2.4 # units from center
model_path = 'model'


# Logging and plotting
do_final_demo = False
show_evolution = True
show_episode_freq = 100
print_freq = 50
pl.close('all')

# Reset TF
tf.reset_default_graph()

# Model definition
model = NNModel(learning_rate=learning_rate)
saver = tf.train.Saver()


#### TRAINING ####
# Logging arrays
state_array = []
negated_action_array = []
reward_array = []
episode_reward_array = []

# Initialization
episode_number = 0
running_reward = None
reward_sum = 0
last_hundred_rewards = q = collections.deque([0.0]*100) # List of 100 zeros to start
solved_cutoff = 195 # taken from openai gym site, 195+ is considered solved for average of last 100 episodes

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  rendering = False

  gradient_buffer = sess.run(model.training_vars)
  for i, grad in enumerate(gradient_buffer):
    gradient_buffer[i] = grad * 0

  #pretty sure observation has this format: [cart position, cart velocity, pole angle, pole rotational velocity]
  observation = env.reset()
  step = 0
  steps_above_threshold = 0
  while (episode_number < max_num_episodes) and (np.mean(last_hundred_rewards) < solved_cutoff):
    #TODO: add additional abort criterion:
    # simulate while episode is not done (episode number will only be incremented, when done)
    step += 1

    x = np.reshape(observation, [1, model.input_dim])
    state_array.append(x)

    # Visualize every show_episode_freq episode
    if show_evolution and episode_number % show_episode_freq == 0:
      rendering = False #was true, but my vm won't run it properly

    if rendering:
      env.render()

    # Get the action to take from the network
    network_output = sess.run(model.action_probability, feed_dict={model.input: x})
    # print(network_output)
    # Introduce randomness during training for exploration purposes
    action = 1 if np.random.rand() < network_output else 0  # network provides the probability of sending 1 as an input
    #Luke: I didn't do the thing below, but I don't think we need it for anything.
    # action_probability_network = #TODO: compute probability of chosen action (!!! is not the same as the network output) # probability of taking the action which was taken
    negated_action_array.append(1 - action)  # dealing with the negated action simplifies the reward function computation

    # Simulate one step of the environment and receive feedback
    observation, reward, done, info = env.step(action)#TODO: conduct 1 step in simulation environment
    reward_sum += reward  # sum up reward for the current episode
    reward_array.append(reward)

    # Commenting these out because the env is designed to handle this already
    # if (observation[2] > max_deflection * 3.14 / 180.0):
    #   print("Tipped over")
    #   done = True
    #
    # if (np.abs(observation[0]) > max_displacement):
    #   print("Cart moved too far")
    #   done = True

    if done:
      # Let's handle the reward checking first
      last_hundred_rewards.appendleft(reward_sum) # using collection since it is O(1) instead of O(N)
      last_hundred_rewards.pop()


      # Only increment episode number if one episode is done
      episode_number += 1

      # prepare training
      episode_x = np.vstack(state_array)
      state_array = []
      negated_episode_actions = np.vstack(negated_action_array)
      negated_action_array = []
      episode_reward = np.vstack(reward_array)
      reward_array = []

      # Compute mean normal discounted reward
      discounted_episode_reward = sup.discountedReward(episode_reward, gamma)
      discounted_episode_reward -= np.mean(discounted_episode_reward)
      discounted_episode_reward /= np.std(discounted_episode_reward)

      # get the gradients for this episode and save it for later (required for batch_gradient)
      gvps_episode = sess.run(model.gradient_variable_pairs,
                              feed_dict={model.input: episode_x,
                                         model.negated_action: negated_episode_actions,
                                         model.reward_signal: discounted_episode_reward})


      episode_reward_array.append(reward_sum)

      # Add gradient to batch gradient
      for i, gvp in enumerate(gvps_episode):
        gradient_buffer[i] += gvp[0]

      # After desired number of episodes (batch_size), update weights
      if episode_number % batch_size == 0:
        sess.run(model.update_grads,
                 feed_dict={model.Wfc1_grad: gradient_buffer[0],
                            model.Wfc2_grad: gradient_buffer[1],
                            model.Wfc3_grad: gradient_buffer[2]})

        # Reset batch gradient
        for i, grad in enumerate(gradient_buffer):
          gradient_buffer[i] = grad * 0


        if reward_sum/batch_size > 10000:
          print('Task solved in {} episodes.'.format(episode_number))
          break

      # reset for next episode
      if episode_number % print_freq == 0:
        print('Reward for last episode {}: {} \t'.format(episode_number, reward_sum))
        print('Mean over last hundred rewards: {}'.format(np.mean(last_hundred_rewards)))
      reward_sum = 0
      step = 0
      observation = env.reset()

      if rendering:
        env.close()
        rendering = False

  if not os.path.exists(model_path):
    os.mkdir(model_path)
  saver.save(sess, model_path + '/model-final.ckpt')
  print('Saved final model.')

  # Visualize once after optimization
  if do_final_demo:
    print('Final demonstration')
    observation = env.reset()
    done= False
    while not done:
      # Compute action
      x = np.reshape(observation, [1, model.input_dim])
      network_output = sess.run(model.action_probability, feed_dict={model.input: x})
      # now don't take a stochastic but deterministic action
      action = 1 if network_output > 0.5 else 0
      # Simulate
      observation, reward, done, _ = env.step(action)
      # env.render()

  pl.figure('Training history')
  ax = pl.gca()
  ax.plot(episode_reward_array)
  ax.grid('on')
  ax.plot([0, len(episode_reward_array)], [200, 200], c='r', lw=0.5)
  ax.set_ylim([0, 210])
  pl.show(block=False)


env.close()
