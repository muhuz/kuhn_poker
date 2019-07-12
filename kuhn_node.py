
class KuhnNode():
    """
    Each Node represents an information set. The node stores informations
    about the valid actions that can be made at this information set, namely
    the cumulative regret of those actions, the strategy profile, and the
    strategy sum which is used to return the average strategy profile, or
    the strategy sum divided by the number of iterations used to train.
    """

    def __init__(self, infoSet):
        self.iterations = 0
        self.card = infoSet[0]
        self.history= infoSet[1]
        self.regretSum = [0, 0]
        self.strategy = [[0.5, 0.5]]
        self.strategySum = [0, 0]

    def update_regretSum(self, a, v):
        self.regretSum[a] += v

    def update_strategySum(self, a, v):
        self.strategySum[a] += v

    def update_strategy(self):
        self.iterations += 1
        total_regret = 0

        # Get total non-negative counterfactual regret
        for regret in self.regretSum:
            if regret > 0:
                total_regret += regret

        if total_regret > 0:
            new_strategy = []
            for i in range(len(self.regretSum)):
                if self.regretSum[i] > 0:
                    new_strategy.append(self.regretSum[i] / total_regret)
                else:
                    new_strategy.append(0.0)
            self.strategy.append(new_strategy)
        else:
            self.strategy.append([0.5 for i in range(2)])


