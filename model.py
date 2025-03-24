import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.w0 = nn.Parameter(53, 100)
        self.b0 = nn.Parameter(1, 100)
        self.w1 = nn.Parameter(100, 1)
        self.b1 = nn.Parameter(1, 1)
        self.parameters = [self.w0, self.b0, self.w1, self.b1]
        self.learning_rate = .9
        self.numTrainingGames = 3000
        self.batch_size = 200

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return (states - Q_target)**2

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        temp = nn.ReLU(nn.AddBias(nn.Linear(states, self.w0), self.b0))
        return nn.AddBias(nn.Linear(temp, self.w1), self.b1)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x,y), [self.w0, self.w1, self.b0, self.b1])
                self.w0.update(grad[0], -0.9)
                self.w1.update(grad[1], -0.9)
                self.b0.update(grad[2], -0.9)
                self.b1.update(grad[3], -0.9)
