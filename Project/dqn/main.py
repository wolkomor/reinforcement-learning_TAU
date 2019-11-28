import gym
import torch.optim as optim

from dqn_model import DQN, DQN_bn, DQN_RAM
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

import argparse

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def main(env, num_timesteps, model_type, save_path, rbs):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=model_type,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=rbs,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        save_path=save_path
    )
    
    
models_dict = {
    "DQN": DQN,
    "BN": DQN_bn,
    "RAM": DQN_RAM,
    }

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    parser = argparse.ArgumentParser(description='DQN args')
    parser.add_argument('--model', default='DQN', choices=['DQN', 'BN', 'RAM'],
                        help='model type: DQN, BN, RAM')
    parser.add_argument('--path', default="statistics.pkl",
                        help='paths to save pickle')
    parser.add_argument('--rbs', default=REPLAY_BUFFER_SIZE,
                        help='replay buffer size')

    args = parser.parse_args()
    
    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps, model_type=models_dict[args.model], save_path=args.path, rbs=args.rbs)
