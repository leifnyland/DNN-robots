import os
import random
from collections import deque
import json
import psutil

import numpy as np
from keras.models import load_model
from rgkit import rg
from rgkit.settings import AttrDict
from rgkit.run import Runner, Options
import rgkit.game as rg_game
from rgkit.settings import settings
import ast
from importlib.util import spec_from_file_location, module_from_spec
import time


import gc
from keras.backend import clear_session
import tensorflow as tf


def initialize_map(map_file):
    """
    Initialize the map file so that the robot can use it.
    @param map_file: path to map file
    @return: None
    """
    with open(map_file) as _:
        settings.init_map(ast.literal_eval(_.read()))


def run_games(player1, player2, map_file, num_games=1, visualize=False, game_seed=None):
    """
    Run one game between player1 and player2 using the map_file.
    @param player1: rgkit.game.Player
    @param player2: rgkit.game.Player
    @param map_file: 'rgkit/maps/default.py'
    'rgkit/maps/afffsdd/{card|castle|field|fourcorners|hourglass|mansion|metasquares|sqaure}.py'
    @param num_games: number of games to run
    @param visualize: whether to show GUI animation of game.
    @param game_seed: the seed for the random number generate to make results repeatable.
    @return: (result, seconds); result = 0 for loss, 0.5 for draw, 1 for win; time to run game.
    """
    t0 = time.time()
    if game_seed is None:
        game_seed = random.randint(0, 2147483647)
    results = Runner(
        players=[player1, player2],
        options=Options(
            map_filepath=map_file,
            game_seed=game_seed,
            n_of_games=num_games,
            headless=not visualize,
            quiet=5,
            print_info=False,
            animate_render=False,
            play_in_thread=False,
            autorun=visualize,
            symmetric=not map_file[:-3].endswith('mansion'),
        )
    ).run()
    seconds = time.time() - t0
    results = np.array(results)
    result = (np.sign(results[:, 0] - results[:, 1]) + 1) / 2
    return result, seconds


def get_function(function_path):
    """
    Load a function from its path.
    :param function_path: 'module.submodule.function_name'
    :return: the Python function
    """
    # https://stackoverflow.com/a/19393328
    import importlib
    module_name, func_name = function_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    f = getattr(module, func_name)
    return f


def get_player(module_name, name):
    """
    Return the rgkit.game.Player and Robot from a module.
    @param module_name: path to the module
    @param name: name to give the player
    @return: the rgkit.game.Player, Robot
    """
    robot = get_function(f'{module_name}.Robot')()
    return rg_game.Player(robot=robot, name=name), robot


class DQLRobot:
    def __init__(self, model_dir=None, model_params=None, training_params=None, compete=False):
        """
        Constructor for DQLRobot.
        @param model_dir: Directory to store this robot. If it exists, load the latest version. If it's a model file,
        load it instead of the latest version.
        @param model_params: a dictionary with hyperparameters to pass to the build_func
        @param training_params: a dictionary with the training hyperparameters
        @param compete: a boolean indicating whether to play with epsilon=0 or to explore with computed epsilon
        """
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.compete = compete
        self.model = None

        # get model_file and model_dir
        model_file = None
        if model_dir is None:
            import pandas as pd
            words = pd.read_csv('eff_large_wordlist.txt', sep='\t', names=['roll', 'word'])['word']
            random_name = '_'.join(words.sample(n=2))
            model_dir = f'my_robots/{random_name}'
        elif model_dir.endswith('.h5'):
            model_dir, model_file = os.path.split(model_dir)
        elif os.path.isdir(model_dir):
            model_file = self.get_latest_model_file(model_dir)

        self.model_dir = model_dir

        if model_file is not None:
            # load existing model
            self.version, self.num_games, self.num_moves = self.parse_model_file(model_file)
            with open(self.params_file, 'r') as fp:
                self.model_params = json.load(fp)
            with open(self.training_params_file, 'r') as fp:
                self.training_params = json.load(fp)
            self.model = load_model(self.model_file)
            self.get_state, self.get_reward, self.build_model = self.load_static_methods(self.robot_file)
        else:
            # create new model in directory
            self.version = self.num_moves = self.num_games = 0
            self.get_state, self.get_reward, self.build_model = self.load_static_methods('robot.py')
            # include input_shape in model_params
            self.model_params.update(dict(input_shape=self.get_state_size()))
            self.model = self.build_model(**self.model_params)
            # save files to the model directory
            os.makedirs(model_dir, exist_ok=False)
            with open(self.params_file, 'w') as fp:
                json.dump(self.model_params, fp)
            with open(self.training_params_file, 'w') as fp:
                json.dump(self.training_params, fp)
            with open('robot.py', 'r') as fp_in:
                with open(self.robot_file, 'w') as fp_out:
                    fp_out.write(fp_in.read())
            with open('robot_helpers.py', 'r') as fp_in:
                with open(self.robot_helpers_file, 'w') as fp_out:
                    fp_out.write(fp_in.read())
            self.save()

        # placeholders to use to call 'act' during the game
        self.location = None
        self.player_id = None
        self.hp = None
        self.robot_id = None

        # storage for computed states, actions for a turn
        self.actions = [{}, {}]
        self.actions_index = [{}, {}]
        self.states = [{}, {}]

        # where to store memoized data to reuse
        self.allies = {}
        self.zombie_allies = {}
        self.all_allies = {}

        # exponential moving average for loss function
        self.ema_alpha = .9
        self.ema_loss = float('nan')

        # bogus state to use as filler
        self.state_size = self.get_state_size()
        self.bogus_state = np.zeros(self.state_size)

        # keep track of average reward per game played in training
        self.episode_rewards = [0.0]
        self.episode_moves = 0
        self.average_reward = float('nan')

        # preallocate numpy arrays to use for training
        batch_size = self.training_params['batch_size']
        state_shape = tuple([batch_size] + [self.state_size])
        self.state = np.zeros(shape=state_shape, dtype=np.float32)
        self.action = np.zeros(shape=batch_size, dtype=np.int64)
        self.reward = np.zeros(shape=batch_size, dtype=np.float32)
        self.next_state = np.zeros(shape=state_shape, dtype=np.float32)
        self.done = np.zeros(shape=batch_size, dtype=bool)

        num_actions = len(rg.actions((10, 10)))
        max_robots = 100
        self.action_values = np.zeros(shape=(max_robots, num_actions))

        buffer_size = self.training_params['buffer_size']
        self.memory = deque(maxlen=buffer_size)
        self.competition_results = []

        # elo rating assumes all contests against same opponent with self.elo + opponent.elo = 2400.
        self.elo = 1200
        self.elo_k = 16

        self.game_start_time = None
        self.timing = []
        self.beaten = set()

    @staticmethod
    def parse_model_file(model_file):
        """
        Return the version, number of games and number of moves from model_file
        @param model_file: the model file name
        @return:
        """
        file_name = os.path.split(model_file[:-3])[-1]
        return tuple(int(token) for token in file_name.split('_')[1:])

    @property
    def model_file(self):
        """
        @return: path to model file
        """
        return os.path.join(self.model_dir, f'model_{self.version}_{self.num_games}_{self.num_moves}.h5')

    @property
    def log_file(self):
        """
        @return: path to log file
        """
        return os.path.join(self.model_dir, f'trainer.log')


    @property
    def training_params_file(self):
        """
        @return: path to training parameters file
        """
        return os.path.join(self.model_dir, f'training_params.json')

    @property
    def params_file(self):
        """
        @return: path to model parameters file
        """
        return os.path.join(self.model_dir, 'model_params.json')

    @property
    def robot_file(self):
        """
        @return: path to robot file
        """
        return os.path.join(self.model_dir, 'robot.py')

    @property
    def robot_helpers_file(self):
        """
        @return: path to robot helpers file
        """
        return os.path.join(self.model_dir, 'robot_helpers.py')

    @property
    def win_percentage(self):
        """
        Return predicted win percentage based on elo rating of self.elo and opponent (2400 - self.elo)
        @return: win percentage
        """
        return 1 / (1 + 10 ** ((2400 - 2*self.elo) / 400))

    def log(self, opponent='', first=False):
        """
        Print and write to log file.
        @param opponent: Name of opponent
        @param first: Whether this is the first entry in the log file
        @return: None
        """
        # temporarily set compete=false to compute current epsilon
        compete = self.compete
        self.compete = False

        # print the next version to save, we are currently training it
        # previous version is the one that is saved
        version = self.version + 1 - int(first)
        memory_usage = psutil.Process().memory_info().rss / 2**20
        competition_result = self.competition_results[-1] if self.competition_results else ' '
        if competition_result == '+' and opponent not in self.beaten:
            print(f'\n BEAT {opponent} for the first time!!!\n')
            self.beaten.add(opponent)

        average_time = float('nan')
        if len(self.timing) > 0:
            average_time = np.mean(self.timing[-20:])
        message = (f'{os.path.split(self.model_dir)[-1]} v{version} {self.num_games} games, {self.num_moves} moves, '
                   f'e={self.epsilon:.3f}, loss: {self.ema_loss:.1f}, {competition_result}, '
                   f'{self.win_percentage:.1%} elo% vs. {opponent}, '
                   f'avg. reward: {self.average_reward:.3f}, '
                   f'{average_time:.3f} sec/game, {np.sum(self.timing)/60:.1f} min. total, '
                   f'{memory_usage:.1f} MB')
        print('\r' + message, end='')
        if first:
            print()
        with open(self.log_file, 'a') as fp:
            fp.write(message + '\n')

        # set compete back to original value
        self.compete = compete

    @property
    def epsilon(self):
        """
        @return: current epsilon to use for exploration
        """
        if self.compete:
            # for competition, don't explore just play best move
            return 0.
        eps_annealing_steps = self.training_params['eps_annealing_steps']
        explore_final_eps = self.training_params['explore_final_eps']
        if self.num_moves > eps_annealing_steps:
            # after annealing keep 'explore_final_eps' for exploration
            return explore_final_eps
        else:
            # use a linear schedule for epsilon
            a = self.num_moves / eps_annealing_steps
            return 1. - a * (1 - explore_final_eps)

    def set_learn(self):
        self.compete = False

    def set_compete(self):
        self.compete = True

    @staticmethod
    def get_latest_model_file(model_dir):
        """
        Return the latest model file in the directory
        @param model_dir: the path to the model directory
        @return: the latest version of the model file
        """
        latest_version = -1
        model_file = None
        for f in os.listdir(model_dir):
            if f.startswith('model_') and f.endswith('.h5'):
                this_version = int(f.split('_')[1])
                if this_version > latest_version:
                    latest_version = this_version
                    model_file = f
        return model_file

    def get_num_games_moves(self, version):
        """
        Return the number of games and moves for this version of the model
        @param version: the model version number
        @return: the number of games, number of moves
        """
        for f in os.listdir(self.model_dir):
            if f.startswith(f'model_{version}_') and f.endswith('.h5'):
                num_games = int(f.split('_')[2])
                num_moves = int(f.split('_')[3][:-3])
                return num_games, num_moves
        raise FileNotFoundError(f'model_{version} does not exist')

    def load_model(self, version=None):
        """
        Load the model with supplied version or the latest model
        @param version: version number to load, latest if None
        @return: the Keras model
        """
        if version is None:
            self.version = self.get_latest_model_file(self.model_dir)
        else:
            self.version = version
        self.num_games, self.num_moves = self.get_num_games_moves(self.version)
        self.model = load_model(self.model_file)

    @staticmethod
    def load_static_methods(robot_file):
        """
        Load the functions from the robot file "robot.py"
        @param robot_file: the path to the robot file
        @return: get_state func, get_reward func, build_model func.
        """
        spec = spec_from_file_location('robot', robot_file)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'get_state'), getattr(module, 'get_reward'), getattr(module, 'build_model')

    def get_state_size(self):
        """
        Figure out the state size using the get_state method
        @return: The state size (integer)
        """
        location = (12, 9)
        robot = AttrDict(location=location, player_id=0, robot_id=0, hp=50)
        # turn must be negative so that it doesn't conflict with self.allies = {turn: [list of allies]}, etc.
        game = AttrDict(turn=-1, robots=AttrDict({location: robot}))
        state = self.get_state(game, robot)
        return state.shape[0]

    def get_allies(self, game):
        """
        Get list of robot allies on the board
        @param game: the game
        @return: list of robot allies
        """
        key = game.turn, self.player_id
        # store this computation to avoid recomputing for the same turn
        if key not in self.allies:
            self.allies = {key: [bot for bot in game.robots.values() if bot.player_id == self.player_id]}
        return self.allies[key]
    
    def get_zombie_allies(self, game):
        """
        Get list of robot "zombie" allies who are no longer on the board.
        @param game: the game
        @return: list of robot allies
        """
        key = game.turn, self.player_id
        if key not in self.zombie_allies:
            self.zombie_allies = {key: [bot for bot in game.zombies.values() if bot.player_id == self.player_id]}
        return self.zombie_allies[key]

    def get_all_allies(self, game):
        """
        Get list of all robot allies including zombies.
        @param game: the game
        @return: list of robot allies
        """
        key = (game.turn, self.player_id)
        if key not in self.all_allies:
            self.all_allies = {key: self.get_allies(game) + self.get_zombie_allies(game)}
        return self.all_allies[key]

    def get_states(self, game, robots):
        """
        Get state for each robot.
        @param game: the game
        @param robots: list of robots
        @return: numpy 2-d array of states (one per row)
        """
        return np.array([self.get_state(game, r) for r in robots])

    def get_actions(self, robots, states):
        """
        Get the actions (robot_game tuples) and the index of each action for the model.
        @param robots: list of robots
        @param states: numpy array of states
        @return: the list of actions, numpy array of action indexes
        """
        robots_actions = [rg.actions(r.location) for r in robots]
        if len(robots_actions) == 0:
            return [], []

        # select random "exploration" bots
        random_mask = np.random.uniform(low=0, high=1, size=len(states)) < self.epsilon

        # initialize empty action values
        action_values = self.action_values[:len(states), :]

        # set random action values
        action_values[random_mask] = np.random.uniform(
            low=0, high=1, size=(np.sum(random_mask), action_values.shape[1])
        )

        if np.sum(~random_mask) > 0:
            # set nonrandom action values from the model
            masked_states = tf.convert_to_tensor(states[~random_mask])
            action_values[~random_mask] = self.model(masked_states).numpy()

        # remove invalid actions
        invalid_mask = np.array([
            [robot_action is None for robot_action in robot_actions] for robot_actions in robots_actions
        ])
        action_values[invalid_mask] = float('-inf')

        # find the best valid action index
        actions_index = np.argmax(action_values, axis=1)

        # pull out the tuple actions
        actions = [robot_actions[action_index] for robot_actions, action_index in zip(robots_actions, actions_index)]
        return actions, actions_index

    def save(self):
        """
        Save the model with the file name containing its version, num games, and num moves
        @return: None
        """
        self.model.save(self.model_file)

    def save_new_version(self):
        """
        Increment the version, save, and reset elo rating
        @return: None
        """
        self.version += 1
        self.save()
        # reset elo, and thereby expected win percentage against previous version
        self.elo = 1200

    def get_rewards(self, game, robots):
        """
        Get the reward for each robot
        @param game: the game
        @param robots: list of robots
        @return: list of rewards
        """
        return [self.get_reward(game, r) for r in robots]

    def remember(self, states, actions, rewards, next_states, dones):
        """
        Store these in the memory
        @param states: a dictionary of {robot_id: state}
        @param actions: a dictionary of {robot_id: action_id}
        @param rewards: a dictionary of {robot_id: reward}
        @param next_states: a dictionary of {robot_id: next_state}
        @param dones: a dictionary of {robot_id: done?}
        @return: None
        """
        robot_ids = list(states.keys() | actions.keys() | rewards.keys() | next_states.keys() | dones.keys())
        for rid in robot_ids:
            if rid in states:
                if rid not in next_states:
                    assert rid in dones, f'robot {rid} has no next_state but is not done.'
                next_state = next_states.get(rid, self.bogus_state)

                # store in memory
                self.memory.append((states[rid], actions[rid], rewards[rid], next_state, dones[rid]))

                self.num_moves += 1
                self.episode_rewards[-1] += rewards[rid]
                self.episode_moves += 1
                train_freq = self.training_params['train_freq']
                if self.num_moves % train_freq == 0:
                    loss, seconds = self.train()
                    if np.isnan(self.ema_loss):
                        self.ema_loss = loss
                    self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * loss
            else:
                # can't learn from a bot being spawned
                assert rid not in actions, f'robot {rid} has no state but has an action.'

    def train(self, epochs=1):
        t0 = time.time()
        loss = float('nan')
        buffer_size = self.training_params['buffer_size']
        batch_size = self.training_params['batch_size']
        gamma = self.training_params['gamma']
        # start training once buffer size is reached
        if len(self.memory) >= buffer_size:
            # pull random batch from memory
            batch = random.sample(self.memory, batch_size)

            # construct state, action, reward, next_state, done matrices
            state, action, reward, next_state, done = list(zip(*batch))
            self.state[:batch_size, :] = state
            self.action[:batch_size] = action
            self.reward[:batch_size] = reward
            self.next_state[:batch_size, :] = next_state
            self.done[:batch_size] = done

            # store in convenient temporary variables
            state = self.state[:batch_size, :]
            action = self.action[:batch_size]
            reward = self.reward[:batch_size]
            next_state = self.next_state[:batch_size, :]
            done = self.done[:batch_size]

            # overwriting reward as value
            value = reward
            not_done_next_state = tf.convert_to_tensor(next_state[~done])
            # q-value update equation
            value[~done] += gamma * self.model(not_done_next_state).numpy().max(axis=1)

            # current predictions
            state = tf.convert_to_tensor(state)
            y_hat = self.model(state).numpy()

            # target updates the particular actions that were taken
            # replace the target for the actions actually taken and leave the others the same
            y_hat[range(len(y_hat)), action] = value

            history = self.model.fit(state, y_hat, batch_size=32, epochs=epochs, verbose=0)
            loss = history.history['loss'][-1]

            # try to keep memory use down
            clear_session()

        return loss, time.time() - t0

    def act(self, game):
        """
        Called by robot game to determine the action for the robot "self"
        @param game: the game
        @return: the action for "self"
        """
        if game.new_game:
            # if it's a new game, do some bookkeeping
            self.game_start_time = time.time()
            self.actions = [{}, {}]
            self.actions_index = [{}, {}]
            self.states = [{}, {}]
            self.allies = {}
            self.zombie_allies = {}
            self.all_allies = {}
            memory_usage = psutil.Process().memory_info().rss
            if memory_usage > 2**30:
                print('garbage collect')
                gc.collect()

        elif game.new_turn:
            # If it's a new turn, compute all turns for teammates and store them
            # It's faster to send all bots through the model than to do so one at a time
            assert game.turn < 101

            # figure out the states, actions, rewards for this turn
            allies = self.get_allies(game)
            all_allies = self.get_all_allies(game)

            # store by robot_id
            ally_ids = [r.robot_id for r in allies]
            all_ally_ids = [r.robot_id for r in all_allies]

            # states and actions from previous turn
            states = self.states[self.player_id]
            actions_index = self.actions_index[self.player_id]

            # rewards and dones determined this turn
            rewards = dict(list(zip(all_ally_ids, self.get_rewards(game, all_allies))))
            dones = dict(list(zip(all_ally_ids, [r.hp <= 0 or game.turn == 99 for r in all_allies])))

            # new state and action for this turn
            next_states = self.get_states(game, allies)
            # model needs numpy array
            next_actions, next_actions_index = self.get_actions(allies, next_states)
            next_actions = dict(list(zip(ally_ids, next_actions)))
            next_actions_index = dict(list(zip(ally_ids, next_actions_index)))

            # store in same dict format
            next_states = dict(list(zip(ally_ids, next_states)))

            # save new actions and states to learn from next turn
            self.actions[self.player_id] = next_actions
            self.actions_index[self.player_id] = next_actions_index
            self.states[self.player_id] = next_states

            if not self.compete:
                # add this info to the memory for training
                self.remember(states, actions_index, rewards, next_states, dones)
        elif game.game_over:
            # If the game is over, do some more bookkeeping
            assert game.turn == 101
            if not self.compete and self.episode_moves > 0:
                # maintain reward for both players
                self.num_games += 1
                self.episode_rewards[-1] /= self.episode_moves
                self.episode_rewards.append(0.0)
                self.episode_moves = 0
                self.average_reward = np.mean(self.episode_rewards[-101:-1])
                self.timing.append(time.time() - self.game_start_time)
                self.game_start_time = None
            if self.compete:
                # check result
                num_allies = num_enemies = 0
                for loc, bot in game.robots.items():
                    if bot.player_id == self.player_id:
                        num_allies += 1
                    else:
                        num_enemies += 1
                result = (np.sign(num_allies - num_enemies) + 1.) / 2.
                r = {0.: '-', 0.5: '.', 1.: '+'}[result]
                self.competition_results.append(r)
                # update elo and expected win percentage
                self.elo += self.elo_k * (result - self.win_percentage)
        else:
            # We should have already computed this robot's move and stored it in self.actions
            robot = game.zombies.get(self.robot_id, game.robots.get(self.location))
            if robot.robot_id in game.zombies:
                return ('guard',)
            action = self.actions[self.player_id][robot.robot_id]
            return action

        return ('guard',)
