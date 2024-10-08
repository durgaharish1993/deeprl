import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Create directory for saving the plots
output_dir = os.path.join(os.getcwd(), 'output/Q_learning')

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Generic Environment class
class GenericEnv:
    def __init__(self, state_space, action_space, terminal_states):
        """
        Initialize the environment.
        state_space: List of all possible states.
        action_space: List of all possible actions.
        terminal_states: List of terminal states.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.terminal_states = terminal_states

    def reset(self):
        """Reset the environment and return the initial state."""
        return self.state_space[0]

    def step(self, state, action):
        """
        Take an action in the environment. Override in subclasses.
        state: Current state
        action: Action to take
        Returns:
        new_state: The new state after taking the action.
        reward: The reward received after taking the action.
        done: Boolean indicating if the new state is terminal.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")


# Q-learning class with integrated plotting
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=0.1):
        """
        Initialize Q-learning.
        env: The environment in which the agent operates.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration-exploitation parameter.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {state: {action: 0.0 for action in env.action_space} for state in env.state_space}
        self.rewards_per_episode = []

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)  # Exploration
        else:
            return max(self.Q[state], key=self.Q[state].get)  # Exploitation

    def update_q_value(self, state, action, reward, next_state, done):
        """Update the Q-value using the Q-learning formula."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state].values())

        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * target

    def train(self, num_episodes):
        """Train the agent using Q-learning."""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.update_q_value(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    break

            self.rewards_per_episode.append(total_reward)
        print(f"Max Episode {np.argmax(self.rewards_per_episode)}: Reward: {np.max(self.rewards_per_episode)}")  # Store total reward for plotting

    def plot_rewards(self):
        """Plot the reward per episode and save it to output folder."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_per_episode, label="Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Progression over Episodes")
        plt.legend()
        plt.grid()

        # Save plot
        plt.savefig(f"{output_dir}/rewards_per_episode.png")
        plt.close()

    def plot_q_values(self):
        """Visualize Q-values and save the heatmap to output folder."""
        states = list(self.Q.keys())

        # Plot state-action values as a heatmap
        heatmap_data = np.zeros((self.env.size, self.env.size))
        for (i, j) in self.Q:
            heatmap_data[i, j] = max(self.Q[(i, j)].values())

        # Convert heatmap data to float type if necessary
        heatmap_data = np.array(heatmap_data, dtype=np.float32)

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Q-value")
        plt.title("State-Action Values (Max Q-Value per State)")

        # Save plot
        plt.savefig(f"{output_dir}/q_values_heatmap.png")
        plt.close()

    def extract_optimal_policy(self):
        """Extract the optimal policy (best action for each state)."""
        policy = {}
        for state in self.Q:
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        print("Optimal policy:", policy)
        return policy

    def plot_optimal_policy(self):
        """Visualize the optimal policy with arrows."""
        policy = self.extract_optimal_policy()

        # Create a 2D grid for visualizing the policy
        direction_map = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        x, y = np.meshgrid(np.arange(self.env.size), np.arange(self.env.size))
        u = np.zeros_like(x, dtype=np.float32)
        v = np.zeros_like(y, dtype=np.float32)

        for (i, j) in policy:
            action = policy[(i, j)]
            dx, dy = direction_map[action]
            u[i, j] = dx
            v[i, j] = dy

        plt.figure(figsize=(6, 6))
        plt.quiver(y, x, u, -v, scale=1, scale_units='xy')
        plt.title("Optimal Policy (Arrow shows best action)")

        # Save plot
        plt.savefig(f"{output_dir}/optimal_policy.png")
        plt.close()


# Example environment: GridWorld
class GridWorld(GenericEnv):
    def __init__(self, size=5):
        state_space = [(i, j) for i in range(size) for j in range(size)]
        action_space = ['up', 'down', 'left', 'right']
        terminal_states = [(size - 1, size - 1)]
        super().__init__(state_space, action_space, terminal_states)
        self.size = size

    def step(self, state, action):
        x, y = state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.size - 1:
            y += 1

        new_state = (x, y)
        reward = 1 if new_state in self.terminal_states else -0.1
        done = new_state in self.terminal_states
        return new_state, reward, done


# Visualization of agent's path through GridWorld
def plot_agent_path(agent, env, num_episodes):
    state = env.reset()
    path = [state]

    for _ in range(num_episodes):
        action = agent.choose_action(state)
        next_state, _, done = env.step(state, action)
        path.append(next_state)
        state = next_state
        if done:
            break

    # Plot the path
    x, y = zip(*path)
    plt.figure(figsize=(6, 6))
    plt.plot(y, x, marker="o", color="blue")
    plt.title("Agent's Path through GridWorld")
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Save plot
    plt.savefig(f"{output_dir}/agent_path.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Initialize the environment and Q-learning agent
    # Initialize the environment and Q-learning agent
    env = GridWorld(size=10)
    agent = QLearning(env, alpha=0.5, gamma=0.99, epsilon=0.1)

    # Train the agent
    num_episodes = 10000
    agent.train(num_episodes)

    # Plot the total rewards per episode
    agent.plot_rewards()

    # Visualize the learned Q-values as a heatmap
    agent.plot_q_values()

    # Visualize the agent's optimal policy using arrows
    agent.plot_optimal_policy()

    # Visualize the agent's path through the environment
    plot_agent_path(agent, env, num_episodes=100)

    print("Training complete. All plots are saved in the 'output/Q_learning' directory.")
