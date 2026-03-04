from rgkit import game as rg_game
from robot_helpers import DQLRobot, get_player, run_games, initialize_map
import time
import pandas as pd
import random as rand


class DQLTrainer:
    def __init__(self, model_dir=None, model_params=None, total_time_steps=int(1e7), explore_fraction=0.1,
                 explore_final_eps=0.01, map_file='rgkit/maps/default.py', compete_against_zoo=False, visualize=False,
                 buffer_size=int(1e4), learning_starts=int(1e5), gamma=0.99, train_freq=128, batch_size=1024):
        """
        Initialize a deep q-learner trainer.
        @param model_dir: The model directory
        @param model_params: Dictionary of model parameters to pass to build_fn
        @param total_time_steps: Number of robot actions to simulate before stopping training
        @param explore_fraction: Fraction of total time steps to use for annealing the exploration epsilon
        @param explore_final_eps: The minimum exploration epsilon value to use
        @param map_file: The path to the map file
        @param compete_against_zoo: Whether to compete against the robot zoo (True) or the previously saved version of
        itself (False)
        @param visualize: Whether to visualize one game when a new version of the robot is saved
        @param buffer_size: The number of robot actions to keep in memory to learn from
        @param learning_starts: The number of robot actions to collect before training the model
        @param gamma: The discount factor for the Q-Value equation
        @param train_freq: Update the weights of the model after every 'train_freq' robot actions
        @param batch_size: The batch size (number of robot moves) for training
        """

        self.total_time_steps = total_time_steps
        self.map_file = map_file
        self.compete_against_zoo = compete_against_zoo
        self.visualize = visualize

        initialize_map(self.map_file)

        eps_annealing_steps = int(round(explore_fraction * total_time_steps))
        training_params = dict(
            eps_annealing_steps=eps_annealing_steps,
            explore_final_eps=explore_final_eps,
            learning_starts=learning_starts,
            train_freq=train_freq,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            map_file=map_file
        )

        # create one deep q-learning robot and two players who use it
        self.robot = DQLRobot(model_dir, model_params, training_params)
        self.player1 = rg_game.Player(robot=self.robot, name=f'v{self.robot.version+1}')
        self.player2 = rg_game.Player(robot=self.robot)


        # load the robot zoo
        self.zoo = []
        if self.compete_against_zoo:
            df = pd.read_csv('round_robin.csv')
            self.zoo = df.sort_values(by='win%', ascending=False)['player1'].tolist()

    def get_next_opponent(self):
        """
        Get the next opponent, either the next best robot from the zoo or the previously save version of player 1
        @return: opponent name, its rgkit.game.Player
        """
        if self.compete_against_zoo and len(self.zoo) > 0:
            opponent = self.zoo.pop()
            opponent_player, opponent_robot = get_player(f'zoo.{opponent}', opponent)
        else:
            opponent = f'v{self.robot.version}'
            opponent_robot = DQLRobot(self.robot.model_dir, compete=True)
            opponent_player = rg_game.Player(robot=opponent_robot, name=opponent)
        return opponent, opponent_player

    def train(self):
        """
        Main method for self-play
        @return: None
        """
        start_time = time.time()

        # get the next opponent
        opponent, opponent_player = self.get_next_opponent()
        # log the first line
        self.robot.log(first=True)

        # trying to learn from btr
        btr_player, btr_robot = get_player(f'zoo.btr', 'btr')
        dulladob_player, dulladob_robot = get_player(f'zoo.dulladob', 'dulladob')
        spferical_player, spferical_robot = get_player(f'zoo.spferical', 'spferical')

        previous_average_reward = float('-inf')
        # for the specified number of robot moves
        while self.robot.num_moves < self.total_time_steps:
            # self-play and learn
            self.robot.set_learn()
            run_games(self.player1, self.player2, self.map_file)
            # rand_num = rand.randint(1, 10)
            # if rand_num == 1:
            #     run_games(self.player1, btr_player, self.map_file)
            # if rand_num == 2:
            #     run_games(self.player1, dulladob_player, self.map_file)
            # if rand_num == 3:
            #     run_games(self.player1, spferical_player, self.map_file)

            # check against previous version
            self.robot.set_compete()
            run_games(self.player1, opponent_player, self.map_file)

            # log another line
            self.robot.log(opponent)

            # check for progress against opponent or previous average reward
            beat_current_opponent = self.robot.win_percentage > 0.7
            improved_avg_reward = self.robot.num_games % 100 == 0 and \
                self.robot.average_reward > previous_average_reward
            if beat_current_opponent or improved_avg_reward:
                # save a new version
                self.robot.save_new_version()
                self.player1._name = f'v{self.robot.version}'
                print(f', Saving version {self.robot.version}')
                if beat_current_opponent:
                    print(f'\n DEFEATED {opponent}!!!\n')
                    if self.visualize:
                        # play a game to visualize (still in compete mode)
                        run_games(self.player1, opponent_player, self.map_file, visualize=True)
                    # update opponent and average reward
                    previous_average_reward = self.robot.average_reward
                    opponent, opponent_player = self.get_next_opponent()

        duration = time.time() - start_time
        print(f'\n{duration/3600:.2f} hours')


def main():
    """
    Main fuction to run trainer.
    @return: None
    """
    model_params = {'learning_rate': 1e-6}

    # map_file = 'rgkit/maps/default.py'
    # DQLTrainer(model_dir="my_robots/skedaddle_puma", model_params=model_params, map_file=map_file, visualize=True, compete_against_zoo=False, batch_size=4096, buffer_size=int(1e5)).train()

    map_file = 'rgkit/maps/afffsdd/mansion.py'
    DQLTrainer(model_dir="my_robots/enrich_mushiness", model_params=model_params, map_file=map_file, visualize=True, compete_against_zoo=False, batch_size=8192, buffer_size=int(4e5)).train()

    # map_file = 'rgkit/maps/afffsdd/field.py'
    # DQLTrainer(model_dir="my_robots/anyplace_passage", model_params=model_params, map_file=map_file, visualize=True, compete_against_zoo=True).train()

    # DQLTrainer(model_params=model_params, map_file=map_file, visualize=True, compete_against_zoo=False, batch_size=4096, buffer_size=int(1e5)).train()


    #wasp emote beats btr 25%

if __name__ == '__main__':
    main()
