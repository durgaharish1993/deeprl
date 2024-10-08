import pygame
import numpy as np

class PongGame:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.reset()
        self.observation_space = (self.width, self.height, 3)  # Assuming RGB
        self.action_space = 3  # 0: No action, 1: Up, 2: Down
        self.clock = pygame.time.Clock()
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

    def reset(self):
        self.ball_pos = [self.width // 2, self.height // 2]
        self.ball_vel = [5, 5]
        self.paddle_pos = [self.height // 2 - 60]  # Centered paddle
        return self.get_observation()

    def step(self, action):
        # Paddle movement based on action
        if action == 1 and self.paddle_pos[0] > 0:  # Up
            self.paddle_pos[0] -= 10
        elif action == 2 and self.paddle_pos[0] < self.height - 120:  # Down
            self.paddle_pos[0] += 10

        # Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Ball collision with top and bottom walls
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.height:
            self.ball_vel[1] *= -1

        # Ball collision with paddle
        if (self.ball_pos[0] <= 30 and
            self.paddle_pos[0] < self.ball_pos[1] < self.paddle_pos[0] + 120):
            self.ball_vel[0] *= -1

        # Reset ball if it goes out of bounds
        done = False
        if self.ball_pos[0] < 0 or self.ball_pos[0] > self.width:
            done = True
            self.reset()

        # Return next state, reward, done
        return self.get_observation(), 1 if not done else -1, done, {}

    def get_observation(self):
        # Create an RGB frame representation of the game
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw paddle
        pygame.draw.rect(frame, (255, 255, 255), (20, self.paddle_pos[0], 10, 120))
        # Draw ball
        pygame.draw.circle(frame, (255, 255, 255), (self.ball_pos[0], self.ball_pos[1]), 10)
        return frame

    def render(self, mode='human'):
        if mode == 'human':
            self.screen.fill((0, 0, 0))
            self.screen.blit(pygame.surfarray.make_surface(self.get_observation()), (0, 0))
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS

    def close(self):
        pygame.quit()
