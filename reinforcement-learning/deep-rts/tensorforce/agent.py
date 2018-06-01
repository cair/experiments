from tensorforce.core.networks import Network
import tensorflow as tf
from tensorforce.agents import PPOAgent, DQNAgent, DDPGAgent


class Agents:

    def __init__(self):

        self.states = {
            "vector": dict(type='float', shape=(7,)),
            "image": dict(type='float', shape=(672, 672, 3))

        }

        self.actions = dict(type='int', num_actions=16)

        self.network = [
            [
                dict(type='input', inputs=['image']),
                dict(type='conv2d', size=2, window=4, stride=1, padding="SAME", activation='relu'),
                dict(type='conv2d', size=2, window=2, stride=1, padding="SAME", activation='relu'),
                dict(type='flatten'),
                dict(type='output', output='out_image'),
            ],
            # [
            #    dict(type='input', inputs=['vector']),
            #    # dict(type='dense', size=6, activation='relu'),
            #    dict(type='output', output='out_vector'),
            # ],
            [
                dict(type='input', inputs=['out_image', 'vector']),
                # dict(type='dense', size=512, activation='relu'),
                dict(type='dense', size=64, activation='relu'),
                dict(type='output', output='prediction'),
            ]
        ]

        self.states_preprocessing = {
            "image": [
                {"type": "image_resize", "width": 64, "height": 64},
                {"type": "grayscale"},
                # {"type": "center"},
                {"type": "sequence", "length": 4}
            ]
        }

        self.summary = dict(directory="./board/",
                            steps=50,
                            labels=[]
                            )
        self.agent = None

        self.agents = {

        }

    def set_active(self, agent_name):
        try:
            try:
                self.agent = self.agents[agent_name]
            except:
                if agent_name == "PPO":
                    self.agents[agent_name] = PPOAgent(
                        states=self.states,
                        actions=self.actions,
                        network=self.network,
                        summarizer=self.summary,
                        states_preprocessing=self.states_preprocessing,
                        batching_capacity=1000,
                        step_optimizer=dict(
                            type='adam',
                            learning_rate=1e-4
                        )
                    )
                    return self.agents[agent_name]
                elif agent_name == "DQN":
                    self.agents[agent_name] = DQNAgent(
                        states=self.states,
                        actions=self.actions,
                        network=self.network,
                        summarizer=self.summary,
                        states_preprocessing=self.states_preprocessing,
                        batching_capacity=1000,
                        optimizer=dict(
                            type='adam',
                            learning_rate=1e-4
                        )
                    )
                    return self.agents[agent_name]

                else:
                    raise RuntimeError("Missing model type! %s" % agent_name)

            return self.agent
        except KeyError:
            print("Could not find agent '%s'." % agent_name)
