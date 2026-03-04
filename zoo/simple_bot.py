from rgkit import rg


class Robot:
    def __init__(self):
        self.location = None
        self.player_id = None
        self.robot_id = None
        self.hp = None

    def act(self, game):
        if game.new_game or game.new_turn or game.game_over or self.hp <= 0:
            return ['guard']
        if self.location == rg.CENTER_POINT:
            return ['guard']

        for loc in rg.locs_around(self.location, filter_out=('obstacle', 'invalid')):
            if loc in game.robots and game.robots[loc].player_id != self.player_id:
                return ['attack', loc]

        return ['move', rg.toward(self.location, rg.CENTER_POINT)]
