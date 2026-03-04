import os
import time
from rgkit import game as rg_game
from robot_helpers import DQLRobot
import random
import pandas as pd
from robot_helpers import get_player, initialize_map, run_games


class DQLCompare:
    def __init__(self, player1_path, player2_path, games_to_evaluate=20, render_game=False,
                 map_file='rgkit/maps/default.py'):
        """
        Setup a series of games between two players.
        @param player1_path: Path to your model directory or to a zoo robot's Python file
        @param player2_path: Path to your model directory or to a zoo robot's Python file
        @param games_to_evaluate: number of games to play against each opponent
        @param render_game: whether to render one game for each opponent
        @param map_file: the path to the map file to use, e.g., "rgkit/maps/default.py"
        """
        self.games_to_evaluate = games_to_evaluate
        self.render_game = render_game
        self.map_file = map_file

        initialize_map(self.map_file)

        players = []
        for player_path in [player1_path, player2_path]:
            if player_path.endswith('.py'):
                # it's a zoo robot
                robot_path = player_path[:-3].replace('/', '.').replace('\\', '.')
                player, robot = get_player(robot_path, robot_path)
            else:
                # it's a DQL bot
                robot = DQLRobot(player_path, model_params=None, compete=True)
                player = rg_game.Player(robot=robot, name=player_path.split('/')[-1])
            players.append(player)

        self.player1, self.player2 = players

    def run(self):
        """
        The main runner method, prints the results and visualizes if desired
        @return:  None
        """
        random.seed()
        results, seconds = run_games(self.player1, self.player2, self.map_file, num_games=self.games_to_evaluate)

        s = ''.join(['+' if x == 1 else '-' if x == 0 else '.' for x in results])
        wins = sum(x == '+' for x in s)
        losses = sum(x == '-' for x in s)
        draws = sum(x == '.' for x in s)
        p = (wins + 0.5*draws) / (wins + losses + draws)

        if p > 0.5:
            msg = f'{self.player1.name()} defeats {self.player2.name()}!'
        elif p < 0.5:
            msg = f'{self.player1.name()} losses to {self.player2.name()}!'
        else:
            msg = f'{self.player1.name()} and {self.player2.name()} draw.'

        print(f'{msg}\n{s}\n({wins}W-{draws}D-{losses}L): {p:.2%}')

        if self.render_game:
            run_games(self.player1, self.player2, self.map_file, visualize=True)


def main():
    player1_path = 'current best/default' #red
    player2_path = 'zoo/btr.py' #red
    # player2_path = 'ColinBots/default' #blue
    map_file = 'rgkit/maps/default.py'
    # map_file = 'rgkit/maps/afffsdd/mansion.py'
    DQLCompare(player1_path=player1_path, player2_path=player2_path, map_file=map_file, render_game=True, games_to_evaluate=10).run()


if __name__ == '__main__':
    main()
