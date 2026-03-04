

def get_reward(game, bot):
    """
    This reward was the first semi successful attempt at a simple, non-tying reward scheme.

    Mansion: It learned faster than it could increase its elo on mansion, and made it all the way to btr
    (~82% on webcat). It actually beat the hard bots the most, but still had some trouble with the lower bots.

    Default: A fast learner, but had trouble making it past simple bot.

    Use the game and the robot to determine the robot's reward for taking its action last turn.
    @param game: the game
    @param bot: the robot
    @return: the robot's reward
    """
    reward = 0

    if bot.damage_caused - bot.damage_taken >= 0:
        reward += bot.damage_caused

    return reward

'''
        This is the reward for my best webcat default (80%) and mansion (83%).
'''

    # damage dealt
    if bot.damage_caused > 0:
        reward += bot.damage_caused

    # damage taken
    if bot.damage_taken > 0:
        reward -= bot.damage_taken

    if bot.kills > 0:
        reward += bot.kills * 20

    # dies
    if died(bot):
        reward -= 25

    # wins
    if game.game_over:
        if bot.teammates > bot.opponents:
            reward += 50000
        elif bot.teammates < bot.opponents:
            reward -= 500

    return reward

## couldn't get past simple

reward = 0

if bot.damage_caused - bot.damage_taken >= 0:
    reward += bot.damage_caused

if bot.damage_taken - bot.damage_caused >= 0:
    reward -= bot.damage_taken

if bot.kills > 0:
    reward += bot.kills * 20

# dies
if died(bot):
    reward -= 25

if game.turn > 85:
    reward += bot.teammates - bot.opponents

if game.game_over:
    if bot.teammates > bot.opponents:
        reward += 50000
    elif bot.teammates < bot.opponents:
        reward -= 500

# state
neighborhood = rg.get_neighborhood(game, robot, within=7, metric='cityblock')
state = [x is not None for x in neighborhood['spawn']]
state += [0 if x is None else x['hp'] / 50 for x in neighborhood['enemies']]
state += [x is not None for x in neighborhood['enemies']]
state += [0 if x is None else x['hp'] / 50 for x in neighborhood['allies']]
state += [x is not None for x in neighborhood['allies']]
state += [x is not None for x in neighborhood['obstacle']]
state += [x is not None for x in neighborhood['invalid']]
state += [spawn_next_turn(game)]
state += [game.turn / 100]
state += [(robot.location[0] - 10) / 10, (robot.location[1] - 10) / 10]
state += [robot.hp / 50]
state = np.array(state, dtype=np.float32)
return state






