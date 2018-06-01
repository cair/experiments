import random


class Random:

    def __init__(self, game, player):
        self.game = game
        self.player = player

    def reset(self):
        pass

    def update(self, seconds):
        action = random.randint(0, len(self.player.action_space) - 1)
        self.player.do_action(action)

    def on_defeat(self):
        pass

    def on_victory(self):
        pass