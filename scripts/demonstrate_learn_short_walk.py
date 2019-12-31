import datetime

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

import tensorflow as tf

from diffrascape.env import ShortWalk
from diffrascape.rl.deep_q_network import (
    TensorboardAgentCallback,
    DeepQNetworkAgent,
    NoisyDense,
)


short_walk_env = ShortWalk(width=20, height=20)

q_network = Sequential()
q_network.add(
    Conv2D(
        input_shape=(short_walk_env.width, short_walk_env.height, 1),
        data_format="channels_last",
        filters=16,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
    )
)
q_network.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
q_network.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
q_network.add(Flatten())
noisy_dense_0 = NoisyDense(
    noise_mean=0.0, noise_std=0.017, units=128, activation="relu"
)
q_network.add(noisy_dense_0)
noisy_dense_1 = NoisyDense(noise_mean=0.0, noise_std=0.017, units=64, activation="relu")
q_network.add(noisy_dense_1)
noisy_dense_2 = NoisyDense(
    noise_mean=0.0, noise_std=0.017, units=4, activation="linear"
)
q_network.add(noisy_dense_2)

q_network.compile(optimizer="adam", loss="mse")

# tensorboard --logdir logs/
training_history_writer = tf.summary.create_file_writer(
    "logs" + f"/dqn_{datetime.datetime.now().strftime('%d%m%Y%H%M')}"
)


class EpisodeEndCallback(TensorboardAgentCallback):
    def on_episode_end(self, episode_n, total_reward, total_loss):
        # print(episode_n)
        # print(total_reward)
        # print(total_loss)
        if episode_n % 100 == 0:
            print(
                f"Episode: {episode_n}, Total Reward: {total_reward:.3f}, Total Loss: {total_loss:.3f}"
            )
            with self.tb_writer.as_default():
                tf.summary.scalar("total reward", total_reward, step=episode_n)
                tf.summary.scalar("total loss", total_loss, step=episode_n)


class SNRCallback(TensorboardAgentCallback):
    def __init__(self, noisy_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers

    def on_episode_end(self, episode_n, total_reward, total_loss):
        if episode_n % 100 == 0:
            with self.tb_writer.as_default():
                for noisy_layer in self.noisy_layers:
                    tf.summary.scalar(f"{noisy_layer.name} SNR", noisy_layer.get_snr(), step=episode_n)


print(q_network.summary())

dqn_agent = DeepQNetworkAgent(
    q_network=q_network,
    batch_size=32,
    history_size=1000,
    gamma=0.99,
    target_q_network_update_interval=100,
)

dqn_agent.train(
    env=short_walk_env,
    nb_episodes=100000,
    callbacks=[
        EpisodeEndCallback(tb_writer=training_history_writer),
        SNRCallback(noisy_layers=(noisy_dense_0, noisy_dense_1, noisy_dense_2), tb_writer=training_history_writer)
    ],
)
