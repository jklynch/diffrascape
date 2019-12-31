import datetime
import sys

import gym
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
import tensorflow as tf

from diffrascape.rl.deep_q_network import DeepQNetworkAgent, NoisyDense, TensorboardAgentCallback

env = gym.make("CartPole-v1")
print(env)
print(f"action space: (shape: {env.action_space.n}) {env.action_space}")
print(f"observation space: (shape: {env.observation_space.shape}) {env.observation_space}")


# batch normalization helps for ~300 episodes and then seems to make things worse
# dropout alone seems to help a tiny bit

q_network = Sequential()
#noisy_layer_1 = NoisyDense(units=64, input_shape=env.observation_space.shape, activation='relu')
q_network.add(Dense(units=128, input_shape=env.observation_space.shape, activation='relu'))
q_network.add(Dropout(rate=0.01))  # not bad
#q_network.add(BatchNormalization())
#noisy_layer_2 = NoisyDense(units=32, activation="relu")
q_network.add(Dense(units=64, activation="relu"))
q_network.add(Dropout(rate=0.01))  # not bad
#q_network.add(BatchNormalization())
q_network.add(Dense(units=32, activation="relu"))
q_network.add(Dropout(rate=0.01))  # not bad
#q_network.add(BatchNormalization())
q_network.add(Dense(units=env.action_space.n, activation="linear"))

q_network.compile(optimizer="adam", loss="mean_squared_error")

print(q_network.summary())


class EpisodeEndCallback(TensorboardAgentCallback):
    def on_episode_end(self, episode_n, **kwargs):
        sys.stdout.write(f"Episode: {episode_n}")
        for name, value in kwargs.items():
            sys.stdout.write(f" {name}: {value:3.3f}")
        sys.stdout.write("\n")
        with self.tb_writer.as_default():
            for name, value in kwargs.items():
                tf.summary.scalar(name, value, step=episode_n)


class SNRCallback(TensorboardAgentCallback):
    def __init__(self, noisy_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers

    def on_episode_end(self, episode_n, **kwargs):
        if episode_n % 100 == 0:
            with self.tb_writer.as_default():
                for noisy_layer in self.noisy_layers:
                    tf.summary.scalar(f"{noisy_layer.name} SNR", noisy_layer.get_snr(), step=episode_n)


# tensorboard --logdir logs/
training_history_writer = tf.summary.create_file_writer(
    "logs" + f"/dqn_{datetime.datetime.now().strftime('%d%m%Y%H%M')}"
)

dqn_agent = DeepQNetworkAgent(
    q_network=q_network,
    batch_size=32,
    history_size=10000,
    min_training_history_size=100,
    gamma=0.95,
    target_q_network_update_interval=1000,
    callbacks=[
        EpisodeEndCallback(tb_writer=training_history_writer),
        #SNRCallback([noisy_layer_1, noisy_layer_2], tb_writer=training_history_writer)
    ],
)

dqn_agent.train(
    env=env,
    nb_episodes=10000,
    begin_epsilon_decay=1000,
    epsilon_decay_steps=1000000
)
