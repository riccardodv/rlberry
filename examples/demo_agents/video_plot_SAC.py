"""

"""

from rlberry.manager import plot_writer_data, AgentManager
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch import SACAgent
import gym


def env_ctor():
    env = PBall2D()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    return env


env_kwargs = dict()

agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget=500,
    n_fit=1,
    enable_tensorboard=True,
)

agent.fit()

# Plot of the cumulative reward.
output = plot_writer_data(agent, tag="loss_q1", title="Loss q1")
output = plot_writer_data(agent, tag="loss_q2", title="Loss q2")
output = plot_writer_data(agent, tag="loss_v", title="Loss critic")
output = plot_writer_data(agent, tag="loss_act", title="Loss actor")
