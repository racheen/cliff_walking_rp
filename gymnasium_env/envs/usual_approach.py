from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class CliffWalker(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=(12,4)):
        self.xsize, self.ysize = size
        self.window_xsize = 3*256
        self.window_ysize = 256

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1,self.ysize-1]), shape=(2,), dtype=int),
                "target": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1,self.ysize-1]), shape=(2,), dtype=int),
            }
        )

        self.cliff_positions = {(i, self.ysize - 1) for i in range(1, self.xsize - 1)}
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.array([0,self.ysize-1]) 
        self._target_location = np.array([self.xsize - 1, self.ysize - 1]) 
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        new_position = np.clip(
            self._agent_location + direction, a_min=np.array([0, 0]), a_max=np.array([self.xsize - 1, self.ysize - 1])
        )

        if any(np.array_equal(new_position, pos) for pos in self.cliff_positions):
            self._agent_location = np.array([0, 0])  
            reward = -100
            terminated = False
        else:
            self._agent_location = new_position
            reward = -1
            terminated = np.array_equal(self._agent_location, self._target_location)


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_xsize, self.window_ysize))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_xsize, self.window_ysize))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_xsize / self.xsize
        )  
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for pos in self.cliff_positions:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(pos),
                    (pix_square_size, pix_square_size),
                ),
            )

        for x in range(self.xsize + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_xsize, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_xsize),
                width=3,
            )
            
        for x in range(self.xsize+1):                 
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_ysize),
                width=3,
            )


        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
