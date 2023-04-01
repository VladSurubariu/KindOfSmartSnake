import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from game_engine import Game as game

env = game()
actions = 4
states = 12

#env = gym.make('CartPole-v0')
#states = env.observation_space.shape[0]
#actions = env.action_space.n

episodes = 10

def first_random_actions():
    for episode in range (0, episodes):
        state = env.reset()
        done = False
        score = 0
        
        while(not done):
            env.render(mode='')
            action = random.choice([0,1,2,3])
            _, reward, done, _= env.step(action)
            score+=reward
            
        print('Episode:{} Score:{}'.format(episode, score))
        
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,24)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model



def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=500, window_length = 1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
    return dqn

first_random_actions()
model = build_model(states, actions)
# model.summary()
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
model.summary()

dqn.fit(env, nb_steps=5000, visualize=False, verbose=1, log_interval=500)

# scores = dqn.test(env, nb_episodes=10, visualize=False)
# print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=5, visualize=True)

dqn.save_weights('dqn_weigths.h5f', overwrite=True)