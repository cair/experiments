
import numpy as np

from examples.tensorforce.agent import Agents
from pyDeepRTS import PyDeepRTS

if __name__ == "__main__":
    agents = Agents()
    agent = agents.set_active("PPO")


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
    action_frequency = 5
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
            s1_features = np.array([
                player1.gold(),
                player1.lumber(),
                player1.sGatheredGold,
                player1.sGatheredLumber,
                player1.sDamageDone,
                player1.sDamageTaken,
                player1.sUnitsCreated
            ])

            action = agent.act({"vector": s1_features, "image": s1_image})
            print(action)
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

            agent.observe(reward=r, terminal=t)






"""

# Get new data from somewhere, e.g. a client to a web app
client = MyClient('http://127.0.0.1', 8080)

# Poll new state from client
state = client.get_state()

# Get prediction from agent, execute
action = agent.act(state)
reward = client.execute(action)

# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)


"""