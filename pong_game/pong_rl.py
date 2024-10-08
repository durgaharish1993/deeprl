import dqn_agent
import game_env


def main():
    game = PongGame()
    input_dim = 4  # State dimension
    output_dim = 2  # Action dimension (up or down)
    agent = DQNAgent(input_dim, output_dim)
    game.reset()

    episodes = 1000  # Number of episodes to train
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}/{episodes}, Score: {game.get_score()}, Total Reward: {total_reward}")

        if done:
            game.reset()

if __name__ == '__main__':
    main()
