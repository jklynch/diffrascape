from collections import deque

from keras import Sequential
from keras import backend as K
import keras.initializers
from keras.layers import Dense

import numpy as np


class ExperienceHistory:
    def __init__(self, history_size, batch_size, state_shape):
        self.batch_size = batch_size

        self.state0_history = deque(maxlen=history_size)
        self.action_history = deque(maxlen=history_size)
        self.reward_history = deque(maxlen=history_size)
        self.done_history = deque(maxlen=history_size)
        self.state1_history = deque(maxlen=history_size)

        # print(f"state_shape: {state_shape}")
        self.state0_batch = np.zeros((batch_size, *state_shape), dtype=np.float)
        self.action_batch = np.zeros((batch_size,), dtype=np.int)
        self.reward_batch = np.zeros((batch_size,), dtype=np.float)
        self.done_batch = np.zeros((batch_size,), dtype=np.bool)
        self.state1_batch = np.zeros((batch_size, *state_shape), dtype=np.float)

    def __len__(self):
        return len(self.state0_history)

    def append(self, state0, action, reward, done, state1):
        self.state0_history.append(state0)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.done_history.append(done)
        self.state1_history.append(state1)

    def sample_batch(self):
        batch_indices = np.random.choice(
            a=len(self.state0_history), size=self.batch_size, replace=False
        )
        # print(f"batch_indices: (dtype: {batch_indices.dtype})\n{batch_indices}")
        # print(f"all_rows: (dtype: {self.all_rows.dtype}):\n{self.all_rows}")
        for i, batch_i in enumerate(batch_indices):
            self.state0_batch[i] = self.state0_history[batch_i]
            self.action_batch[i] = self.action_history[batch_i]
            self.reward_batch[i] = self.reward_history[batch_i]
            self.done_batch[i] = self.done_history[batch_i]
            self.state1_batch[i] = self.state1_history[batch_i]

        return (
            self.state0_batch,
            self.action_batch,
            self.reward_batch,
            self.done_batch,
            self.state1_batch,
        )


class PrioritizedExperienceHistory:
    def __init__(
        self,
        priority_alpha,
        initial_priority_beta,
        priority_beta_steps,
        history_size,
        batch_size,
        state_shape,
        rng=None,
    ):
        self.priority_alpha = priority_alpha
        self.initial_priority_beta = initial_priority_beta
        self.priority_beta_steps = priority_beta_steps

        self.priority_beta = initial_priority_beta
        self.priority_beta_inc = (1.0 - initial_priority_beta) / priority_beta_steps

        self.history_size = history_size
        self.batch_size = batch_size
        self.state_shape = state_shape

        # index to write next experience
        self.history_i = 0
        self.history_length = 0

        # initialize first element to 1.0 so the first time we call self.priority.max() we get 1.0
        self.priority = np.zeros((history_size,), dtype=np.float)
        self.priority[0] = 1.0

        self.batch_indices = np.zeros((batch_size,), dtype=np.int)
        self.weights = np.zeros((history_size,), dtype=np.float)

        # keep track of steps internally
        self.step_count = int(0)

        self.state0_history = np.zeros((history_size, *state_shape), dtype=np.float)
        self.action_history = np.zeros((history_size,), dtype=np.int)
        self.reward_history = np.zeros((history_size,), dtype=np.float)
        self.done_history = np.zeros((history_size,), dtype=np.bool)
        self.state1_history = np.zeros((history_size, *state_shape), dtype=np.float)

        self.state0_batch = np.zeros((batch_size, *state_shape), dtype=np.float)
        self.action_batch = np.zeros((batch_size,), dtype=np.int)
        self.reward_batch = np.zeros((batch_size,), dtype=np.float)
        self.done_batch = np.zeros((batch_size,), dtype=np.bool)
        self.state1_batch = np.zeros((batch_size, *state_shape), dtype=np.float)

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __len__(self):
        return self.history_length

    def append(self, state0, action, reward, done, state1):
        self.state0_history[self.history_i] = state0
        self.action_history[self.history_i] = action
        self.reward_history[self.history_i] = reward
        self.done_history[self.history_i] = done
        self.state1_history[self.history_i] = state1

        self.priority[self.history_i] = self.priority.max()

        self.history_i = (self.history_i + 1) % self.state0_history.shape[0]
        self.history_length = min(self.history_length + 1, self.state0_history.shape[0])

    def sample_batch(self):
        priority_prob = np.power(
            self.priority[: self.history_length], self.priority_alpha
        )
        priority_prob /= priority_prob.sum()

        self.batch_indices[0:] = self.rng.choice(
            a=self.history_length, size=self.batch_size, replace=False, p=priority_prob
        )
        self.state0_batch[0:] = self.state0_history[self.batch_indices]
        self.action_batch[0:] = self.action_history[self.batch_indices]
        self.reward_batch[0:] = self.reward_history[self.batch_indices]
        self.done_batch[0:] = self.done_history[self.batch_indices]
        self.state1_batch[0:] = self.state1_history[self.batch_indices]

        self.weights = (self.history_length * priority_prob[self.batch_indices]) ** (
            -1.0 * self.priority_beta
        )
        self.weights /= self.weights.max()

        return (
            self.state0_batch,
            self.action_batch,
            self.reward_batch,
            self.done_batch,
            self.state1_batch,
        )

    def update_batch_priorities(self, batch_loss):
        self.priority[self.batch_indices] = self.weights * (batch_loss + 1e-10)
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_inc)
        self.step_count += 1


class AgentCallback:
    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_episode_begin(self, episode_n):
        pass

    def on_episode_end(self, episode_n, **kwargs):
        pass

    def on_action_begin(self, action):
        pass

    def on_action_end(self, action, reward, done):
        pass

    def on_step_begin(self, step_n):
        pass

    def on_step_end(self, step_n):
        pass

    def on_update_begin(self):
        pass

    def on_update_end(self):
        pass


class TensorboardAgentCallback(AgentCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer


class AgentCallbackGroup(AgentCallback):
    def __init__(self, callback_group):
        self.callback_group = callback_group

    def on_train_begin(self, *args, **kwargs):
        for c in self.callback_group:
            c.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for c in self.callback_group:
            c.on_train_end(*args, **kwargs)

    def on_episode_begin(self, episode_n):
        for c in self.callback_group:
            c.on_episode_begin(episode_n)

    def on_episode_end(self, episode_n, **kwargs):
        for c in self.callback_group:
            c.on_episode_end(episode_n=episode_n, **kwargs)

    def on_action_begin(self, action):
        for c in self.callback_group:
            c.on_action_begin(action=action)

    def on_action_end(self, action, reward, done):
        for c in self.callback_group:
            c.on_action_end(action=action, reward=reward, done=done)

    def on_step_begin(self, step_n):
        for c in self.callback_group:
            c.on_step_begin(step_n=step_n)

    def on_step_end(self, step_n):
        for c in self.callback_group:
            c.on_step_end(step_n=step_n)

    def on_update_begin(self):
        for c in self.callback_group:
            c.on_update_begin()

    def on_update_end(self):
        for c in self.callback_group:
            c.on_update_end()


class DeepQNetworkAgent:
    def __init__(
        self,
        q_network,
        batch_size,
        history_size,
        min_training_history_size,
        gamma,
        target_q_network_update_interval,
        callbacks=None,
    ):
        self.training_q_network = q_network
        self.target_q_network = Sequential.from_config(
            q_network.get_config(), custom_objects={"NoisyDense": NoisyDense}
        )
        self.batch_size = batch_size
        self.history_size = history_size
        self.min_training_history_size = min_training_history_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.target_q_network_update_interval = target_q_network_update_interval
        if callbacks is None:
            callbacks = []
        self.callback_group = AgentCallbackGroup(callback_group=callbacks)

    def train(self, env, nb_episodes, begin_epsilon_decay):
        self.callback_group.on_train_begin()

        experience_history = PrioritizedExperienceHistory(
            priority_alpha=0.6,
            initial_priority_beta=0.4,
            priority_beta_steps=100000,
            history_size=self.history_size,
            batch_size=self.batch_size,
            state_shape=env.observation_space.shape,
        )

        try:
            # use an explicit loop for episodes
            all_step_n = 0
            for episode_n in range(nb_episodes):
                self.callback_group.on_episode_begin(episode_n)

                episode_loss = 0.0
                episode_reward = 0.0

                done = False

                state0 = None
                state1 = env.reset()

                episode_step_n = 0
                while not done:
                    self.callback_group.on_step_begin(episode_step_n)

                    state0 = state1

                    if np.random.rand() <= self.epsilon:
                        action = env.action_space.sample()
                    else:
                        # add the batch dimension to state0
                        q_values = self.training_q_network.predict(
                            np.expand_dims(a=state0, axis=0)
                        )
                        action = np.argmax(q_values)

                    if all_step_n > begin_epsilon_decay:
                        self.epsilon = max(0.02, self.epsilon - 0.0001)

                    self.callback_group.on_action_begin(action)
                    state1, reward, done, _ = env.step(action)
                    self.callback_group.on_action_end(action, reward, done)
                    # if nb_max_episode_steps and episode_step_n >= nb_max_episode_steps - 1:
                    #    # Force a terminal state.
                    #    done = True

                    experience_history.append(
                        state0=state0,
                        action=action,
                        reward=reward,
                        done=done,
                        state1=state1,
                    )
                    if len(experience_history) >= self.min_training_history_size:
                        self.callback_group.on_update_begin()
                        loss, loss_per_sample = self.double_update_q_network(
                            *experience_history.sample_batch()
                        )
                        # print(loss.history)
                        episode_loss += loss
                        experience_history.update_batch_priorities(loss_per_sample)
                        self.callback_group.on_update_end()

                    if all_step_n % self.target_q_network_update_interval == 0:
                        print(f"step {all_step_n}: update target_q_network weights")
                        self.target_q_network.set_weights(
                            self.training_q_network.get_weights()
                        )

                    episode_reward += reward

                    self.callback_group.on_step_end(episode_step_n)
                    episode_step_n += 1
                    all_step_n += 1

                log = {
                    "episode_steps": episode_step_n,
                    "total_loss": episode_loss,
                    "loss_per_step": (episode_loss / episode_step_n),
                    "epsilon": self.epsilon,
                    "beta": experience_history.priority_beta,
                }
                validation_episode_interval = 100
                if episode_n % validation_episode_interval == 0:
                    validation_steps_per_episode = self.validate(
                        env, total_validation_episodes=100
                    )
                    log.update(
                        {
                            "validation_mean_steps_per_episode": np.mean(
                                validation_steps_per_episode
                            )
                        }
                    )

                self.callback_group.on_episode_end(episode_n, **log)

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        self.callback_group.on_train_end()

    def one_network_update_q_network(
        self, state0_batch, action_batch, reward_batch, done_batch, state1_batch
    ):
        """
        DQN with one network.

        This is how to select the q_values corresponding to actions
        >>> q_values = np.random.rand(32, 4)
        >>> q_values
        array([
            [0.3639796 , 0.04337626, 0.63545211, 0.05987049],
            [0.11635481, 0.07105451, 0.51701382, 0.21515642],
            [0.00382548, 0.50653288, 0.06358575, 0.98176547],
            ...
        ])
        >>> actions = np.random.randint(low=0, high=3, size=32)
        >>> actions
        array([1, 2, 2, ...])

        >>> q_values[0:, actions]
        array([0.04337626, 0.51701382, 0.06358575, ...])
        """

        verbose = False
        if verbose:
            print(f"update_q_network")
            print(f"  state0_batch (shape: {state0_batch.shape}):\n{state0_batch}")
            print(f"  action_batch (shape: {action_batch.shape}): {action_batch}")
            print(f"  state1_batch shape: {state1_batch.shape}")

        q_state0_batch = self.training_q_network.predict_on_batch(state0_batch)
        if verbose:
            print(
                f"  q_state0_batch (shape: {q_state0_batch.shape}):\n{q_state0_batch}"
            )

        q_state1_batch = self.training_q_network.predict_on_batch(state1_batch)
        if verbose:
            print(
                f"  q_state1_batch (shape: {q_state1_batch.shape}):\n{q_state1_batch}"
            )
        best_action_state1_batch = np.argmax(q_state1_batch, axis=1)
        if verbose:
            print(
                f"  best_action_state1_batch (shape: {best_action_state1_batch.shape}): {best_action_state1_batch}"
            )
            print(f"  done_batch (shape: {done_batch.shape}):\n{done_batch}")
        # if a step is a terminal step the reward is just 'reward'
        q_state1_batch[done_batch, best_action_state1_batch[done_batch]] = 0.0
        if verbose:
            print(
                f"  q_state1_batch (shape: {q_state1_batch.shape}):\n{q_state1_batch}"
            )
        # update Q only for the taken action in each step
        all_rows = np.arange(q_state0_batch.shape[0])
        expected_discounted_reward_batch = q_state0_batch.copy()
        expected_discounted_reward_batch[all_rows, action_batch] = (
            self.gamma * q_state1_batch[all_rows, best_action_state1_batch]
            + reward_batch
        )
        if verbose:
            print(
                f"  expected_discounted_reward_batch (shape: {expected_discounted_reward_batch.shape}):\n{expected_discounted_reward_batch}"
            )

        loss = self.training_q_network.train_on_batch(
            x=state0_batch, y=expected_discounted_reward_batch
        )

        return loss

    def two_networks_update_q_network(
        self, state0_batch, action_batch, reward_batch, done_batch, state1_batch
    ):
        """
        DQN with two networks.
        :param state0_batch: ndarray with shape (batch size, *state.shape)
          e.g. [[state0] [state1] ...]
        :param action_batch: ndarray with shape (batch size, ) and integral dtype suitable for indexing
          e.g. [action0 action1 ...] or more specifically [1 0 2... ]
        :param reward_batch:
        :param done_batch:
        :param state1_batch:
        :return:

        This is how to select the q_values corresponding to actions
        >>> q_values = np.random.rand(32, 4)
        >>> q_values
        array([
            [0.3639796 , 0.04337626, 0.63545211, 0.05987049],
            [0.11635481, 0.07105451, 0.51701382, 0.21515642],
            [0.00382548, 0.50653288, 0.06358575, 0.98176547],
            ...
        ])
        >>> actions = np.random.randint(low=0, high=3, size=32)
        >>> actions
        array([1, 2, 2, ...])

        >>> q_values[0:, actions]
        array([0.04337626, 0.51701382, 0.06358575, ...])
        """
        # print(f"update_q_network")
        # print(f"  state0_batch shape: {state0_batch.shape}")
        # print(f"  action_batch (shape: {action_batch.shape}): {action_batch}")

        q_training_state0_batch = self.training_q_network.predict_on_batch(state0_batch)
        # print(
        #     f"  q_value_state0_batch (shape: {q_value_state0_batch.shape}):\n{q_value_state0_batch}"
        # )

        # pick the q_values corresponding to the actions
        best_action_state0_batch = np.argmax(q_training_state0_batch, axis=1)
        # print(
        #     f"  argmax_q_value_state0_batch (shape: {best_action_state0_batch.shape}): {best_action_state0_batch}"
        # )

        q_target_state1_batch = self.target_q_network.predict_on_batch(state1_batch)
        # print(
        #     f"  q_value_state1_batch (shape: {q_value_state1_batch.shape}):\n{q_value_state1_batch}"
        # )
        best_action_state1_batch = np.argmax(q_target_state1_batch, axis=1)
        # print(
        #     f"  argmax_q_value_state1_batch (shape: {best_action_state1_batch.shape}): {best_action_state1_batch}"
        # )
        # print(f"  done_batch (shape: {done_batch.shape}): {done_batch}")
        # if a step is a terminal step the reward is just 'reward'
        q_target_state1_batch[done_batch, best_action_state1_batch[done_batch]] = 0.0
        # update Q only for the best action in each step
        all_rows = np.arange(q_training_state0_batch.shape[0])
        expected_discounted_reward_batch = q_training_state0_batch.copy()
        expected_discounted_reward_batch[all_rows, action_batch] = (
            self.gamma * q_target_state1_batch[all_rows, best_action_state1_batch]
            + reward_batch
        )

        loss = self.training_q_network.train_on_batch(
            x=state0_batch, y=expected_discounted_reward_batch
        )

        return loss

    def double_update_q_network(
        self, state0_batch, action_batch, reward_batch, done_batch, state1_batch
    ):
        """
        DQN with two networks.
        :param state0_batch: ndarray with shape (batch size, *state.shape)
          e.g. [[state0] [state1] ...]
        :param action_batch: ndarray with shape (batch size, ) and integral dtype suitable for indexing
          e.g. [action0 action1 ...] or more specifically [1 0 2... ]
        :param reward_batch:
        :param done_batch:
        :param state1_batch:
        :return:

        This is how to select the q_values corresponding to actions
        >>> q_values = np.random.rand(32, 4)
        >>> q_values
        array([
            [0.3639796 , 0.04337626, 0.63545211, 0.05987049],
            [0.11635481, 0.07105451, 0.51701382, 0.21515642],
            [0.00382548, 0.50653288, 0.06358575, 0.98176547],
            ...
        ])
        >>> actions = np.random.randint(low=0, high=3, size=32)
        >>> actions
        array([1, 2, 2, ...])

        >>> q_values[0:, actions]
        array([0.04337626, 0.51701382, 0.06358575, ...])
        """
        # print(f"update_q_network")
        # print(f"  state0_batch shape: {state0_batch.shape}")
        # print(f"  action_batch (shape: {action_batch.shape}): {action_batch}")

        q_training_state0_batch = self.training_q_network.predict_on_batch(state0_batch)
        # print(
        #     f"  q_value_state0_batch (shape: {q_value_state0_batch.shape}):\n{q_value_state0_batch}"
        # )

        # pick the q_values corresponding to the actions
        best_action_state0_batch = np.argmax(q_training_state0_batch, axis=1)
        # print(
        #     f"  argmax_q_value_state0_batch (shape: {best_action_state0_batch.shape}): {best_action_state0_batch}"
        # )

        q_target_state1_batch = self.target_q_network.predict_on_batch(state1_batch)
        # print(
        #     f"  q_value_state1_batch (shape: {q_value_state1_batch.shape}):\n{q_value_state1_batch}"
        # )
        best_action_state1_batch = np.argmax(q_target_state1_batch, axis=1)
        # print(
        #     f"  argmax_q_value_state1_batch (shape: {best_action_state1_batch.shape}): {best_action_state1_batch}"
        # )
        # print(f"  done_batch (shape: {done_batch.shape}): {done_batch}")
        # if a step is a terminal step the reward is just 'reward'
        q_target_state1_batch[done_batch, best_action_state1_batch[done_batch]] = 0.0
        # update Q only for the best action in each step
        all_rows = np.arange(q_training_state0_batch.shape[0])
        expected_discounted_reward_batch = q_training_state0_batch.copy()
        expected_discounted_reward_batch[all_rows, action_batch] = (
            self.gamma * q_target_state1_batch[all_rows, best_action_state0_batch]
            + reward_batch
        )

        # calculate loss per sample before training
        # it is a little risky to call .loss() this way because it could be a list
        loss_per_sample = self.training_q_network.loss_functions[0](
            y_pred=self.training_q_network.predict_on_batch(x=state0_batch),
            y_true=expected_discounted_reward_batch,
        )

        loss = self.training_q_network.train_on_batch(
            x=state0_batch, y=expected_discounted_reward_batch
        )

        return loss, loss_per_sample

    def validate(self, env, total_validation_episodes):
        validation_steps_per_episode = []
        for validation_episode_n in range(total_validation_episodes):
            state = env.reset()
            done = False
            step_n = 0
            while not done:
                # print(f"state: {state}")
                q_values = self.target_q_network.predict(
                    np.expand_dims(a=state, axis=0)
                )
                action = np.argmax(q_values)
                state, reward, done, _ = env.step(action)
                step_n += 1
            validation_steps_per_episode.append(step_n)
        return validation_steps_per_episode


class NoisyDense(Dense):
    """
    Reference: https://arxiv.org/pdf/1706.10295v1.pdf

    output = (mu^w + sigma^w * epsilon^w) @ input + (mu^b + sigma^b * epsilon^b)

    mu^w additive kernel
    sigma^w multiplicative kernel -- use the kernel defined in Dense
    epsilon^w random noise

    mu^b additive bias
    sigma^b multiplicative bias -- use the bias defined in Dense
    epsilon^b random noise

    """

    def __init__(self, noise_std=0.017, **kwargs):
        self.noise_mean = 0.0
        self.noise_std = noise_std

        self.kernel_multiplier = None
        self.bias_multiplier = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        init_value = 0.4 / np.sqrt(np.product(input_shape[1:-2]))

        self.kernel_multiplier = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.RandomUniform(
                minval=(-1.0 * init_value), maxval=init_value
            ),
            name="kernel_multiplier",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias_multiplier = self.add_weight(
                shape=(self.units,),
                initializer=keras.initializers.RandomUniform(
                    minval=(-1.0 * init_value), maxval=init_value
                ),
                name="bias_multiplier",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        super().build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        epsilon_w = K.random_normal(
            shape=self.kernel.shape, mean=self.noise_mean, stddev=self.noise_std
        )
        noisy_kernel = self.kernel + (epsilon_w * self.kernel_multiplier)
        noisy_output = K.dot(inputs, noisy_kernel)
        if self.use_bias:
            epsilon_b = K.random_normal(
                shape=self.bias.shape, mean=self.noise_mean, stddev=self.noise_std
            )
            noisy_bias = self.bias + (epsilon_b * self.bias_multiplier)
            noisy_output = K.bias_add(
                noisy_output, noisy_bias, data_format="channels_last"
            )
        if self.activation is not None:
            noisy_output = self.activation(noisy_output)
        return noisy_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # "noise_mean": self.noise_mean,
                "noise_std": self.noise_std
            }
        )
        return config

    def get_snr(self):
        return np.sqrt(np.mean(np.square(self.kernel))) / (
            np.sqrt(np.mean(np.square(self.kernel_multiplier)))
        )
