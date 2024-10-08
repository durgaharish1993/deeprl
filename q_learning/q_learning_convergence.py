import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Create directory for saving the plots
output_dir = os.path.join(os.getcwd(), 'output/Q_learning_convergence')

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generic Environment class
class GenericEnv:
    def __init__(self, state_space, action_space, terminal_states):
        self.state_space = state_space
        self.action_space = action_space
        self.terminal_states = terminal_states

    def reset(self):
        return self.state_space[0]

    def step(self, state, action):
        raise NotImplementedError("This method should be overridden in the subclass.")

# Q-learning class with integrated plotting
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=0.1, convergence_threshold=0.1, patience=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.convergence_threshold = convergence_threshold  # New parameter for convergence
        self.patience = patience  # Number of episodes to check for convergence
        self.Q = {state: {action: 0.0 for action in env.action_space} for state in env.state_space}
        self.rewards_per_episode = []
        self.max_changes_per_episode = []  # Track maximum changes for convergence plot
        self.last_max_change = float('inf')  # Initialize last maximum change

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)  # Exploration
        else:
            return max(self.Q[state], key=self.Q[state].get)  # Exploitation

    def update_q_value(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state].values())

        # Calculate change in Q-value
        change = abs(self.Q[state][action] - target)
        self.last_max_change = max(self.last_max_change, change)

        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * target

    def train(self, num_episodes):
        convergence_count = 0  # Counter for convergence checks
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            self.last_max_change = 0  # Reset change for this episode

            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.update_q_value(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    break

            self.rewards_per_episode.append(total_reward)
            self.max_changes_per_episode.append(self.last_max_change)  # Store max change for this episode

            # Check for convergence
            if self.last_max_change < self.convergence_threshold:
                convergence_count += 1
                if convergence_count >= self.patience:
                    print(f"Converged after {episode + 1} episodes.")
                    break
            else:
                convergence_count = 0  # Reset if not converged

        print(f"Max Episode {np.argmax(self.rewards_per_episode)}: Reward: {np.max(self.rewards_per_episode)}")

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_per_episode, label="Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Progression over Episodes")
        plt.legend()
        plt.grid()

        plt.savefig(f"{output_dir}/rewards_per_episode.png")
        plt.close()

    def plot_max_changes(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.max_changes_per_episode, label="Max Q-value Change per Episode", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Max Q-value Change")
        plt.title("Convergence Plot: Max Q-value Change over Episodes")
        plt.axhline(y=self.convergence_threshold, color='r', linestyle='--', label='Convergence Threshold')
        plt.legend()
        plt.grid()

        plt.savefig(f"{output_dir}/convergence_plot.png")
        plt.close()

    def plot_q_values(self):
        states = list(self.Q.keys())
        heatmap_data = np.zeros((self.env.size, self.env.size))
        for (i, j) in self.Q:
            heatmap_data[i, j] = max(self.Q[(i, j)].values())

        heatmap_data = np.array(heatmap_data, dtype=np.float32)

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Q-value")
        plt.title("State-Action Values (Max Q-Value per State)")

        plt.savefig(f"{output_dir}/q_values_heatmap.png")
        plt.close()

    def extract_optimal_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        print("Optimal policy:", policy)
        return policy

    def plot_optimal_policy(self):
        policy = self.extract_optimal_policy()

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

    plt.savefig(f"{output_dir}/agent_path.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    env = GridWorld(size=10)
    agent = QLearning(env, alpha=0.5, gamma=0.99, epsilon=0.1)

    num_episodes = 10000
    agent.train(num_episodes)

    agent.plot_rewards()
    agent.plot_max_changes()  # New convergence plot
    agent.plot_q_values()
    agent.plot_optimal_policy()
    plot_agent_path(agent, env, num_episodes=100)

    print("Training complete. All plots are saved in the 'output/Q_learning' directory.")
