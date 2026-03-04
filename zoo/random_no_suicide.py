import random
from rgkit import rg


class Robot:
    def act(self, game):
        if game.new_game or game.new_turn or game.game_over or self.hp <= 0:
            return ['guard']
        locs_around = rg.locs_around(self.location, filter_out=('obstacle', 'invalid'))

        actions = [[a, loc] for a in ['move', 'attack'] for loc in locs_around] + [['guard']]
        return random.choice(actions)
