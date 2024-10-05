import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DiagonalGaussianPolicy,self).__init__()

        #define the mean - neural network layers
        self.mean_network = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        #define the log standard deviation - neural network layers
        self.log_std_network = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
         mu = self.mean_network(state)
         log_std = self.log_std_network(state)

         return mu, log_std

    def sample(self, state):

        mu, log_std = self.forward(state)
        std = torch.exp(log_std) #same shape as std and fills it with random numbers sampled from a standard normal distribution (mean 0, standard deviation 1).
        z = torch.randn_like(std)
        action = mu + std * z

        return  action, mu, std

    def log_likelihood(self, actions, state):
         mu, log_std = self.forward(state)
         std = torch.exp(log_std)
         var = std ** 2

         #calculating the log-likelihood using the gaussion formula
         log_likelihood = -0.5 * (torch.sum((actions - mu) ** 2 / var, dim=-1) +
                                  torch.sum(2 * log_std, dim=-1) +
                                  actions.size(1) * torch.log(torch.tensor(2 * torch.pi)))
         return log_likelihood

    def get_action(self,state):

        action, _,_ = self.sample(state)
        return action

    def get_mean(self,state):
        mu,_ = self.forward(state)
        return mu

    def get_std(self,state):

        _, log_std = self.forward(state)
        return torch.exp(log_std)


if __name__ == "__main__":

    batch_dim = 3
    state_dim = 4
    action_dim = 2
    hidden_dim = 64

    diagonal_policy = DiagonalGaussianPolicy(state_dim,action_dim,hidden_dim)

    #example state input
    state = torch.randn(batch_dim,state_dim)
    print(state)

    #sample an action
    action, mu, std = diagonal_policy.sample(state)
    print("sample action", action)

    #calculate the log likelihood of the sampled action
    log_likelihood_prob = diagonal_policy.log_likelihood(action,state)
    print("log likelihood action", log_likelihood_prob)

    mean_action = diagonal_policy.get_mean(state)
    print("Mean action:", mean_action)

    # Get standard deviation
    std_dev = diagonal_policy.get_std(state)
    print("Standard deviation:", std_dev)












