import argparse
import os
from distutils.util import strtobool
import time
import random
import tempfile
import numpy as np
from typing import Union
import gym
import torch


from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

#TODO: implement a callback system to get the best agent
# Replay generator:
"""
Function to generate a replay of the environment for viewing a trained model

model: trained model
eval_env: environment used to evaluate the agent
video_length: length of video in timesteps
prefix: name prefix for the replay
is_deterministic: deterministic or stochastic

source: https://github.com/huggingface/huggingface_sb3/blob/e1014f8b7195a00e74fa39e2ff1576e64a0cc675/huggingface_sb3/push_to_hub.py#L19

"""
def generate_replay(
    model: BaseAlgorithm,
    eval_env: Union[VecEnv, gym.Env],
    video_length: int,
    prefix: str,
    is_deterministic: bool
):

    # Autowrap, so we only have VecEnv afterward
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    cwd = os.getcwd()
    #create videos directory if needed
    os.makedirs(f"{cwd}/videos", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        env = VecVideoRecorder(
            eval_env,
            tmpdirname,
            record_video_trigger=lambda x: x==0,
            video_length=video_length,
            name_prefix=prefix
        )

        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        try:
            for _ in range(video_length + 1):
                action, lstm_states = model.predict(
                    observation=obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=is_deterministic,
                )
                obs, _, episode_starts, _ = env.step(action)

            env.close()

            inp = env.video_recorder.path           
            out = os.path.join(cwd, f"videos/{prefix}-replay.mp4")
            os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}".format(inp,out))

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(str(e))
            print("Unable to create video replay of model.")

# Argument parser
# Each argument receives: a name, type, default value, and help text
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.splitext(os.path.basename(__file__))[0],
        help="The name of the experiment")
    parser.add_argument('--gym-id', type=str, default='LunarLander-v2',
        help="The gym id")
    parser.add_argument('--h-optimize', type=bool, default=False,
        help="Toggle to optimize hyperparameters")
    parser.add_argument('--algorithm', type=str, default='PPO', 
        help="The algorithm to use for the experiment")
    parser.add_argument('--policy', type=str, default='MlpPolicy', 
        help="The policy to use for the experiment")
    parser.add_argument('--seed', type=int, default=1, 
        help="The seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=25000, 
        help="Total timesteps of the experiment")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="If toggled, cuda will not be enabled by default")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, 
        help="Whether to capture videos of the agents performances (will be in the 'videos' folder)")
    parser.add_argument('--debug-messages', type=bool, default=False,
        help="Toggle to enable debug messages")

    # Algorithm specific arguments
    parser.add_argument('--update-epochs', type=int, default=4, 
        help="The K epochs to update the policy")
    parser.add_argument('--num-envs', type=int, default=4, 
        help="number of parallel game envirnoments")
    parser.add_argument('--num-steps', type=int, default=128, 
        help="Number of steps to run in each envirnoment per policy rollout")
    parser.add_argument('--num-minibatches', type=int, default=4, 
        help="number of mini-batches")
    parser.add_argument('--num-timesteps', type=int, default=1e6, 
        help="The number of timesteps to run the experiment for")

    # Evalute or Train
    parser.add_argument('--load-model', type=str, default="model", 
        help="The model to evaluate")
    # Parse the arguments in the Argument Parser
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}-{args.algorithm}_{args.seed}_{int(time.time())}"

    # TODO: add in StableBaselines3 Tensorboard integration
    # TODO: work on understanding seeding
    random.seed(args.seed) #random seed for our experiment
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set our device to the GPU if the flag is set
    # TODO: work on cuda vs cpu
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Create an environment with a specific seed and id
    #env = gym.make(args.gym_id)
    envs = make_vec_env(args.gym_id, args.num_envs, args.seed)
    # test environment action and observation space

    if args.debug_messages:
        # Observation space consists of two points of feedback
        # 1: horizontal distance to next pipe
        # 2: distance between players y and next holes y
        print("env.observation_space.shape: ", envs.observation_space.shape)
        # Action space consists of two actions
        # 1: jump the bird
        # 2: do nothing
        print("env.action_space.n: ", envs.action_space.n)

    # Get an observation
    start_time = time.time()
    obs = envs.reset()
    num_updates = args.total_timesteps // args.batch_size

    # TODO: implement hyperparamemter optimizations

    #Train/evaluate the model
    # TODO: implement video replay
    if args.load_model == "model":
        model = PPO(policy=args.policy,
                    env=envs,
                    n_steps=args.num_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.update_epochs,
                    verbose=1)
        model.learn(total_timesteps=args.num_timesteps)
        model.save(run_name)    
    else:
        print("Evaluating policy: ", args.load_model)
        model = PPO.load(args.load_model)

    # Evaluate environment
    eval_env = gym.make(args.gym_id)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    # Capture video of the environment
    if args.capture_video:
        generate_replay(model, eval_env, 500, args.load_model, True)


    eval_env.close()
    envs.close()
    
