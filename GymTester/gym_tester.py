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

import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_param_importances

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor

import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics from the run and logs them to Tensorboard
        """
        super().__init__()

    def _on_training_start(self) -> None:
        # Hyperparameter dictionary
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }

        # Metrics dictionary
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }

        # Log the hyperparameters to Tensorboard
        self.logger.record(
            "hparams",
            HParam(hparam_dict=hparam_dict, metric_dict=metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

# The objective function is a function that receives an Optuna trial as an input
def objective(trial, envs, eval_env):
    # dictionary containing hyperparameters to be tuned
    params = {
        "n_epochs": trial.suggest_int("n_epochs", 2, 6),
        "gamma": trial.suggest_float("gamma", 0.9900, 0.9999),
        "total_timesteps": trial.suggest_int("total_timesteps", 500_000, 2_000_000)
    }

    model, score = run_PPO_training(params, envs, eval_env, verbose=0)
    return score

# Function to run the training on the environment with the PPO algorithm
# temporary test function
def run_PPO_training(params, envs, eval_env, verbose=0):
    model = PPO(policy='MlpPolicy',
                env=envs,
                n_steps=1024,
                batch_size=64,
                n_epochs=params['n_epochs'], # parameter to be optimized
                gamma=params['gamma'], # parameter to be optimized
                gae_lambda=0.98,
                ent_coef=0.01,
                verbose=verbose)

    model.learn(total_timesteps=params['total_timesteps']) # parameter to be optimized.
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    score = mean_reward - std_reward

    return model, score


# Function to make a file directory for the current experiment
def create_file_structure():
    cwd = os.getcwd()
    path = f"{cwd}\{args.gym_id}\{args.algorithm}"
    os.makedirs(path, exist_ok=True)
    return path

# Replay generator:
"""
Function to generate a replay of the environment for viewing a trained model

model: trained model
eval_env: environment used to evaluate the agent
video_length: length of video in timesteps
prefix: name prefix for the replay
save_location: file path for the save location of the replay generator
is_deterministic: deterministic or stochastic

source: https://github.com/huggingface/huggingface_sb3/blob/e1014f8b7195a00e74fa39e2ff1576e64a0cc675/huggingface_sb3/push_to_hub.py#L19

"""
def generate_replay(
    model: BaseAlgorithm,
    eval_env: Union[VecEnv, gym.Env],
    video_length: int,
    prefix: str,
    save_location: str,
    is_deterministic: bool
):

    # Autowrap, so we only have VecEnv afterward
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    #create videos directory if needed
    os.makedirs(f"{save_location}/videos", exist_ok=True)

    # This codeblock is to create a temporary directory to save a lot of data we do not need from the
    # video generation process.
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
            out = os.path.join(save_location, f"videos/{prefix}-replay.mp4")
            # This line calls ffmpeg and outputs the video to the desired folder
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {inp} -vcodec h264 {out}".format(inp,out))

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(str(e))
            print("Unable to create video replay of model.")

# Argument parser
# Each argument receives: a name, type, default value, and help text
def parse_args():
    parser = argparse.ArgumentParser()
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
        help="The total number of timesteps for the given experiment")

    # Evalute or Train
    parser.add_argument('--load-model', type=str, default="model", 
        help="The model to evaluate")
    parser.add_argument('--verbose', type=bool, default=True, 
        help="Verbose training output toggle")
    # Parse the arguments in the Argument Parser
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}-{args.algorithm}_{args.seed}_{int(time.time())}"
    video_name = f"{args.seed}-{int(time.time())}"
    cwd = create_file_structure()

    # This section of code sets up the seed for experiment. This is essential for repeatability
    random.seed(args.seed) #random seed for our experiment (used to generate the same random numebrs from experiment to experiment)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set our device to the GPU if the flag is set
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Create an environment with a specific seed and id
    envs = make_vec_env(args.gym_id, args.num_envs, args.seed)
    eval_env = Monitor(gym.make(args.gym_id))
    # test environment action and observation space

    if args.debug_messages:
        # print statement to show the shape of the observation space
        print("env.observation_space.shape: ", envs.observation_space.shape)
        # print statement to show the shape of the action space
        print("env.action_space.n: ", envs.action_space.n)

    # Get an observation
    start_time = time.time()
    obs = envs.reset()

    #Train/evaluate the model
    #Two options here, either we optimize using optuna or we do a training with manual hyperparameters
    if args.h_optimize:
        study = optuna.create_study(sampler=TPESampler(), study_name="test", storage='sqlite:///test.db', direction="maximize", load_if_exists=True)
        study.optimize(lambda trial: objective(trial, envs, eval_env), n_trials=2)
        #plot_param_importances(study)
    
        print("Best trial score:", study.best_trial.values)
        print("Best trial hyperparameters:", study.best_trial.params)

        # Figure out what this output looks like
        model, score = run_PPO_training(study.best_trial.params, envs, eval_env, verbose=1)
        model.save(f"{cwd}/{run_name}")
     
    else:
        if args.load_model == "model":
            model = PPO(policy=args.policy,
                        env=envs,
                        n_steps=args.num_steps,
                        batch_size=args.batch_size,
                        n_epochs=args.update_epochs,
                        verbose=1,
                        tensorboard_log=f"{cwd}\logs")
            model.learn(total_timesteps=args.num_timesteps, tb_log_name="run", callback=HParamCallback())
            model.save(f"{cwd}/{run_name}")    
        else:
            print("Evaluating policy: ", args.load_model)
            model = PPO.load(args.load_model)

    # Evaluate environment  
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    # Capture video of the environment
    if args.capture_video:
        generate_replay(model=model,
                        eval_env=eval_env,
                        video_length=500,
                        prefix=video_name,
                        save_location=cwd,
                        is_deterministic=True)


    eval_env.close()
    envs.close()
    
