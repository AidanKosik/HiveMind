from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.tests import utils

from absl.testing import absltest

from src.HiveMind import HiveMind


class TestHiveEasy(utils.TestCase):
    steps = 200
    step_mul = 16

    def test_hivemind(self):
        with sc2_env.SC2Env(
            map_name="CyberForest",
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=84,
                    minimap=64)),
            step_mul=self.step_mul,
            game_steps_per_episode=self.steps * self.step_mul) as env:
            agent = HiveMind()
            run_loop.run_loop([agent], env, self.steps)
                
        #self.assertLessEqual(agent.episodes, agent.reward)
        # self.assertEqual(agent.steps, self.steps)


if __name__ == "__main__":
    absltest.main()