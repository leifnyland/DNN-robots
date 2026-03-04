import os
import time
from rgkit import game as rg_game
from robot_helpers import DQLRobot
import random
import pandas as pd
from robot_helpers import get_player, initialize_map, run_games


class DQLTournament:
    def __init__(self, model_dir, opponent_dir, games_to_evaluate=20, render_game=False, map_file='rgkit/maps/default.py'):
        """
        Initialize a tournament.
        @param model_dir: the path to the model directory
        @param opponent_dir: the path to a directory containing reflex agent robots (zoo)
        @param games_to_evaluate: number of games to play against each opponent
        @param render_game: whether to render one game for each opponent
        @param map_file: the path to the map file to use, e.g., "rgkit/maps/default.py"
        """
        self.games_to_evaluate = games_to_evaluate
        self.render_game = render_game
        self.map_file = map_file

        initialize_map(self.map_file)

        self.robot1 = DQLRobot(model_dir, model_params=None, compete=True)
        self.player1 = rg_game.Player(robot=self.robot1, name=model_dir.split('/')[-1])
        self.opponents = [f'{opponent_dir}.{f[:-3]}' for f in os.listdir(opponent_dir) if f.endswith('.py')]

    def run(self):
        """
        The main runner method, prints and saves a CSV of results against each opponent
        @return:  None
        """
        t0 = time.time()
        data = []
        random.seed()
        for opponent in self.opponents:
            player2, robot2 = get_player(opponent, opponent)
            results, seconds = run_games(self.player1, player2, self.map_file, num_games=self.games_to_evaluate)

            s = ''.join(['+' if x == 1 else '-' if x == 0 else '.' for x in results])
            wins = sum(x == '+' for x in s)
            losses = sum(x == '-' for x in s)
            draws = sum(x == '.' for x in s)
            p = (wins + 0.5*draws) / (wins + losses + draws)

            print(f'Opponent: {opponent}\n{s}\n({wins}W-{draws}D-{losses}L): {p:.2%}')

            if self.render_game:
                run_games(self.player1, player2, self.map_file, visualize=True)

            data.append({
                'robot': self.robot1.model_file,
                'opponent': opponent,
                'win': wins,
                'loss': losses,
                'draw': draws,
                'percent': 100*p,
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by=['percent'], ascending=False)
        sums = df[['win', 'loss', 'draw']].sum()
        index = len(df)
        df.loc[index] = sums
        p = 100 * (sums @ [1, 0, 0.5]) / sums.sum()
        df.loc[index, 'percent'] = p
        csv_file = f'{self.robot1.model_file}_tournament_{p:.1f}.csv'.replace('/', '.')
        df.to_csv(csv_file, index=False)
        print(df.to_string())
        print(f'{time.time() - t0:.1f} seconds')


def main():
    model_dir = 'ColinBots/field'
    opponent_dir = 'zoo'
    # map_file = 'rgkit/maps/default.py'
    # map_file = 'rgkit/maps/afffsdd/mansion.py'
    map_file = 'rgkit/maps/afffsdd/field.py'
    # map_file = 'rgkit/maps/afffsdd/metasquares.py'
    DQLTournament(model_dir=model_dir, opponent_dir=opponent_dir, map_file=map_file).run()


if __name__ == '__main__':
    main()
