import numpy as np

from kuhn_node import KuhnNode

class KuhnTrainer():

    def __init__(self):
        self.cards = [1, 2, 3]
        self.initialize_regret()
        self.action_map = {0:'p', 1:'b'}

    """
    There are 12 different information sets in Kuhn Poker.

    Active Player has 1, 2, or 3 and there is no history (Active Player going first)
    Active Player has 1, 2, or 3 and history is 'bet' (Active Player going second)
    Active Player has 1, 2, or 3 and history is 'pass' (Active Player going second)
    Active Player has 1, 2, or 3 and history is 'pass bet' (Active Player going first)

    At each of these information sets we come up with a chance sampled strategy.
    The sampled strategy will be initialized so that all valid actions have equal
    probability of being chosen. In this case there are only 2 choices at every
    information set, a player can either bet or pass.
    """

    def shuffle_cards(self):
        np.random.shuffle(self.cards)

    def initialize_regret(self):
        """
        We will represent the different information sets as a tuple containing
        the card the player is holding and the history up until the player
        has to make a decision.

        For example:
        The information set for a player has a 1 and history so far has been 'bet'
        is (1, 'b').
        """
        info_sets = [(1, ''), (2, ''), (3, ''),
                (1, 'b'), (1, 'p'), (2, 'b'),
                (2, 'p'), (3, 'b'), (3, 'p'),
                (1, 'pb'), (2, 'pb'), (3, 'pb')]

        self.InfoDict = {}

        for i in info_sets:
            self.InfoDict[i] = KuhnNode(i)

    def isTerminal(self, h):
        """
        Checks if the history of actions is terminal
        """
        return h in ['pp', 'bb', 'pbp', 'bp', 'pbb']

    def utility(self, i, h):
        """
        Get the utility of the terminal state for player i assuming that
        player 1 always goes first.
        """
        # Cases where the higher card wins
        opp = 1 - i
        if h[-1] == 'b':
            if self.cards[i] > self.cards[opp]:
                return 2
            else:
                return -2
        elif h == 'pp':
            if self.cards[i] > self.cards[opp]:
                return 1
            else:
                return -1
        # Cases where the last better wins
        else:
            if len(h) == 3:
                if i == 0:
                    return -1
                else:
                    return 1
            else:
                if i == 0:
                    return 1
                else:
                    return -1

    def cfr(self, h, i, t, p0, p1):
        """
        This function will train two no-regret algorithms to play
        against each other thus approximating a Nash Equilibrium
        for the game.

        h = the history of actions

        i = the learning player

        t = time step

        p0 = the reach probabilities for player 0

        p1 = the reach probablities for player 1
        """
        if self.isTerminal(h):
            return self.utility(i, h)

        # initialize the value of playing current profile
        # as well as the counterfactual value of playing
        # a at the current information set.
        profile_value = 0
        cf_value = [0, 0]

        # Get the current player and the Node for the corresponding
        # information set
        current_player = len(h) % 2
        card = self.cards[current_player]
        for a, p in enumerate(self.InfoDict[(card, h)].strategy[-1]):
            new_history = h + self.action_map[a]
            if current_player == 0:
                cf_value[a] = self.cfr(new_history, i, t, p0 * p, p1)
            else:
                cf_value[a] = self.cfr(new_history, i, t, p0, p1 * p)
            profile_value += cf_value[a] * p

        # Calculate the regret and update the data in the information set Node.
        if current_player == i:
            if i == 0:
                for a, p in enumerate(self.InfoDict[(card, h)].strategy[-1]):
                    self.InfoDict[(card, h)].update_regretSum(a, p1 * (cf_value[a] - profile_value))
                    self.InfoDict[(card, h)].update_strategySum(a, p0 * self.InfoDict[(card, h)].strategy[-1][a])
                self.InfoDict[(card, h)].update_strategy()
            else:
                for a, p in enumerate(self.InfoDict[(card, h)].strategy[-1]):
                    self.InfoDict[(card, h)].update_regretSum(a, p0 * (cf_value[a] - profile_value))
                    self.InfoDict[(card, h)].update_strategySum(a, p1 * self.InfoDict[(card, h)].strategy[-1][a])
                self.InfoDict[(card, h)].update_strategy()
        return profile_value

    def train_cfr(self, steps):
        for t in range(steps):
            for i in range(2):
                self.shuffle_cards()
                self.cfr('', i, t, 1, 1)

if __name__ == '__main__':
    trainer = KuhnTrainer()
    trainer.train_cfr(200000)
    nodes = trainer.InfoDict
    strategies = {}
    for key, node in nodes.items():
        print("Strategy for Information Set: {}".format(key))
        strategies[key] = np.mean(np.array(node.strategy), axis=0)
        print(strategies[key])
        print(strategies[key].sum())






