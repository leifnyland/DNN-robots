import numpy as np
import keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from rgkit import rg

"""
0 <= game.turn < 101
game.turn == 0 --> spawn new robots during this turn
game.turn % 10 == 0 --> span new robots on these turns
game.turn == 99 --> last turn of the game
game.turn == 100 --> determine reward for the last turn based on resulting state
game.game_over --> 

robot contains following fields:
- hp: (health points [0, 50]) <= 0 means it died after last turn
- location: tuple (row, column)
- player_id: 0 or 1, indicating which team it is on, the bot supplied by get_state and get_reward is on your team.
- teammates: number of teammates this turn
- opponents: number of opponents this turn
- previous_teammates: number of teammates last turn
- previous_opponents: number of opponents last turn
- damage_caused: damage caused by this robot last turn
- damage_taken: damage taken by this robot last turn
- kills: number of opponents this robot killed last turn
- birthturn: True/False whether this is the first turn for this bot
- team_deaths: number of team deaths last turn
- opponent_deaths: number of opponent deaths last turn
"""


def spawn_next_turn(game):
    return game.turn % 10 == 0


def last_turn(game):
    return game.turn == 99


def game_over(game):
    return game.turn == 100


def died(robot):
    return robot.hp <= 0


def get_state(game, robot):
    """
    Determine the "state" of the robot in the game. Robots in the same state should act in the same way
    @param game: the game
    @param robot: the robot
    @return: the state of the robot in the game (numpy vector)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """
    neighborhood = rg.get_neighborhood(game, robot, within=7, metric='cityblock')
    state = [x is not None for x in neighborhood['spawn']]
    state += [x is not None for x in neighborhood['enemies']]
    state += [x is not None for x in neighborhood['allies']]
    state += [x is not None for x in neighborhood['obstacle']]
    state += [x is not None for x in neighborhood['invalid']]
    state += [spawn_next_turn(game)]
    state += [game.turn / 100]
    state += [(robot.location[0] - 10) / 10, (robot.location[1] - 10) / 10]
    state += [robot.hp/50]
    state = np.array(state, dtype=np.float32)
    return state


def build_model(input_shape, learning_rate=0.001):
    """
    Build and compile a model that takes the state as input and estimates q-values for each action index.
    @param input_shape: the size of the state vector
    @param learning_rate: learning rate
    @return: the compiled model
    """
    model = keras.Sequential([
        Input(shape=input_shape),
        Dense(units=800, activation='relu'),
        Dense(units=200, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=10, activation='linear'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def get_reward(game, bot):
    """
    Use the game and the robot to determine the robot's reward for taking its action last turn.
    @param game: the game
    @param bot: the robot
    @return: the robot's reward
    """
    reward = 0

    neighborhood = rg.get_neighborhood(game, bot, within=7, metric='cityblock')

    reward += sum([x is not None for x in neighborhood['allies']])
    reward += bot.damage_caused
    reward += bot.opponent_deaths * 5
    if bot.teammates > bot.opponents:
        reward += (bot.teammates - bot.opponents)
    return reward
