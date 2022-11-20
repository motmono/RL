import numpy as np
import os
import tempfile
import gym
from typing import Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import (DummyVecEnv, VecVideoRecorder, VecEnv)

"""
Function to generate a replay of the environment for viewing a trained model

model: trained model
eval_env: environment used to evaluate the agent
video_length: length of video in timesteps
is_deterministic: deterministic or stochastic

source: https://github.com/huggingface/huggingface_sb3/blob/e1014f8b7195a00e74fa39e2ff1576e64a0cc675/huggingface_sb3/push_to_hub.py#L19

"""

def generate_replay(
    model: BaseAlgorithm,
    eval_env: Union[VecEnv, gym.Env],
    video_length: int,
    is_deterministic: bool
):

    # Autowrap, so we only have VecEnv afterward
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdirname:
        env = VecVideoRecorder(
            eval_env,
            tmpdirname,
            record_video_trigger=lambda x: x==0,
            video_length=video_length,
            name_prefix=""
        )

        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        try:
            for _ in range(video_length + 1):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=is_deterministic,
                )
                obs, _, episode_starts, _ = env.step(action)

            env.close()

            inp = env.video_recorder.path
            out = os.path.join(cwd, "replay.mp4")
            os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}".format(inp,out))

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(str(e))
            print("Unable to create video replay of model.")

