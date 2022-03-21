import gym
import multiworld
import gin
import numpy as np
from types import MethodType

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from multiworld.core.flat_goal_env import FlatGoalEnv

def load_multiworld_env(env_id):
  multiworld.register_all_envs()
  gym_env = FlatGoalEnv(
    gym.make(env_id),
    append_goal_to_obs=True,
  )
  # gym_env.observation_space = gym_env.obs_box
  return tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(gym_env, max_episode_steps=100))

def load(env_id):
  """Creates the training and evaluation environment.

  Args:
    env_name: (str) Name of the environment.
  Returns:
    tf_env, eval_tf_env, obs_dim: The training and evaluation environments.
  """
  tf_env = load_multiworld_env(env_id)
  eval_tf_env = load_multiworld_env(env_id)
  assert len(tf_env.envs) == 1
  assert len(eval_tf_env.envs) == 1
  return (tf_env, eval_tf_env, tf_env.envs[0].observation_space.shape[0] // 2)