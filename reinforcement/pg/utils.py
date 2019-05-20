import os
import tensorflow as tf
from datetime import datetime

def create_directories(env_name, agent_name, base_dir = "."):
    """Create a directories to save logs and checkpoints.

        E.g. ./checkpoints/LunarLander/dqn-vanilla/5-14-2019-9:37:25.43/

        # Arguments
        - env_name (string): name of Gym environment.
        - agent_name (string): name of learning algorithm.
    """

    now = datetime.today()
    date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
    ckpt_dir = "/".join([base_dir, "checkpoints", env_name, agent_name, date_string])
    log_dir = "/".join([base_dir, "logs", env_name, agent_name, date_string])
    os.makedirs(ckpt_dir)
    os.makedirs(log_dir)
    return ckpt_dir, log_dir

def create_metadata(path, *kwargs):
    """Create a metadata file describing checkpoint

        The file should contain all of the arguments passed to the training method.

        You can also include the Gym environment name and agent type to be sure.
    """

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def find_latest_checkpoint(load_path, prefix):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`

        E.g. ./checkpoints/dqn-vanilla-CartPole-v0-GLOBAL_STEP would use find_latest_checkpoint('./checkpoints/', 'dqn-vanilla-CartPole-v0')
    """
    files = os.listdir(load_path)
    matches = [f for f in files if f.find(prefix) == 0]  # files starting with prefix
    max_steps = np.max(np.unique([int(m.strip(prefix).split('.')[0]) for m in matches]))
    latest_checkpoint = load_path + prefix + '-' + str(max_steps)
    return latest_checkpoint
