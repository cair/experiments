
import numpy as np
from tensorforce.agents import DQNAgent

from pyDeepRTS import PyDeepRTS

if __name__ == "__main__":

    agent = DQNAgent(
        states={
            "image": dict(type='float', shape=(672, 672, 3))
        },
        actions=dict(type='int', num_actions=16),
        network=[
            dict(type='conv2d', size=32, window=8, stride=4, padding="SAME", activation='relu'),
            dict(type='conv2d', size=64, window=4, stride=2, padding="SAME", activation='relu'),
            dict(type='conv2d', size=64, window=3, stride=1, padding="SAME", activation='relu'),
            dict(type='flatten'),
            dict(type='dense', size=512, activation='relu')
        ],
        states_preprocessing={
            "image": [
                {"type": "image_resize", "width": 80, "height": 80},
                {"type": "grayscale"},
                # {"type": "center"},
                {"type": "sequence", "length": 1}
            ]
        },
        summarizer=dict(directory="./board/",
                        steps=50,
                        labels=['configuration',
                                'gradients_scalar',
                                'regularization',
                                'inputs',
                                'losses',
                                'variables']
                        ),
        batching_capacity=1000,
        optimizer=dict(
            type='adam',
            learning_rate=1e-4
        )
    )
    config = PyDeepRTS.DeepRTS.Config()
    config.set_instant_town_hall(False)
    g = PyDeepRTS("21x21-2v2.json", config=config, pomdp=False, simple=False)
    player1 = g.add_player()
    player2 = g.add_player()

    g.set_agent_player(player1)

    g.set_max_fps(1000000)
    g.set_max_ups(1000000)
    g.render_every(1)
    g.capture_every(1)
    g.view_every(60)
    g.start()

    ####################################################################
    #
    # Game Rules
    #
    ####################################################################
    action_frequency = 15
    num_games = 1000000

    ####################################################################
    #
    # Agent settings
    #
    ####################################################################
    s0 = None

    for episode in range(num_games):

        counter = 0

        while True:
            counter += 1

            # Render the game state
            g.render()
            g.view()
            g.caption()
            g.gui.capture(save=True)

            # Get and preprocess the state
            s1_image = g.get_state(True)
            s1_image = s1_image.reshape((1, 672, 672, 3))
            action = agent.act({"image": s1_image})

            print(s1_image.shape)
            player1.do_action(action)
            player2.do_action(np.random.randint(0, 16))

            # Process game
            for _ in range(action_frequency):
                g.tick()
                g.update()

            r = -0.01
            t = False

            if g.is_terminal():
                g.reset()
                print("Game %s - %s" % (episode, counter))

                if player1.is_defeated():
                    r = -1
                else:
                    r = 1

                t = True

            print(action, r)
            agent.observe(reward=r, terminal=t)