import logging
import random
from enum import Enum
from typing import Dict, Tuple

import gymnasium
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygame
from prettytable import PrettyTable

from utils.logger import logger

logger.setLevel(logging.INFO)


class Act2048(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def print_action(action: Act2048):
    print(f"Action: {action.name} (int value: {action.value})")


class SGAI_2048(gymnasium.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "cli", "rl-headless"],
        "render_fps": 4,
    }
    """
    This class represents a 2048 game. This is the most straight forward and step by step way I was
    able to find. There are many other ways online which all seemed more complicated to me. Each
    action is broken out specifically with any matrix rotations. Each is broken down into a nice set
    of for loops where the main work is done with indexing the loops. Understanding this may require
    a handwritten example or two of the loops tracking values or a rubber ducky to master.
    NOTE: With a keen eye and knowledge of 2048, you will notice these collapse rules are a bit more...
    collapse-y. This is on purpose to make it more difficult to think ahead. Let me know how you
    think this may impact strategy.
    Scoring is classic 2048: each merge scores the value of the tile it creates, and a move
    that changes nothing on the board costs no_move_penalty points. A new tile spawns every
    turn there is room for one. The game ends when the board is full and nothing can merge.
    """

    def __init__(self, render_mode=None, num_rows: int = 4, num_cols: int = 4):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board = np.zeros((num_rows, num_cols), dtype=int)
        self._add_new()
        self._add_new()  # start with two random tiles
        self.total_score = 0
        self.game_over = False
        self.highest_tile = int(self.board.max())
        self.step_merge_score = 0  # merge points scored by the current step's slide
        self.no_move_penalty = -10  # reward for a move that changes nothing
        self.action_space = gymnasium.spaces.Discrete(len(Act2048))
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=2 ** (num_rows * num_cols + 1),
            shape=[num_rows, num_cols],
            dtype=int,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # references needed for human rendering only
        self.window_size = (400, 500)  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.font_ = None

    def _get_obs(self):
        # copy, not reference: the slide methods mutate self.board in place, so
        # returning the live array would alias every previously returned obs to
        # the current board state (corrupting e.g. RL replay buffers)
        return self.board.copy()

    def _get_info(self, action: str) -> Dict:
        return {
            "board": self.board,
            "action": action,
            "total score": self.total_score,
            "highest tile": self.highest_tile,
            "game over": self.game_over,
            "render mode": self.render_mode,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self._add_new()
        self._add_new()  # start with two random tiles
        self.total_score = 0
        self.game_over = False
        self.highest_tile = int(self.board.max())
        self.step_merge_score = 0
        observation = self._get_obs()
        info = self._get_info("first_turn")
        self.render()
        return observation, info

    def step(self, action: int) -> Tuple:

        action_enum = Act2048(action)
        board_before = self.board.copy()
        self.step_merge_score = 0  # the slides below add each merged tile's value here

        # TODO HW: Is there a better way to implement this sliding and collapsing? (think symmetry)
        if action_enum is Act2048.UP:
            self._slide_up()

        if action_enum is Act2048.DOWN:
            self._slide_down()

        if action_enum is Act2048.LEFT:
            self._slide_left()

        if action_enum is Act2048.RIGHT:
            self._slide_right()

        board_changed = not np.array_equal(board_before, self.board)

        # A new tile spawns every turn there is room for one, even after a move
        # that changed nothing
        if np.any(self.board == 0):
            self._add_new()
        self._is_game_over()

        # Classic 2048 scoring: each merge is worth the value of the tile it creates,
        # and a move that leaves the board unchanged is penalized
        # TODO HW: the reward used to be the change in the total tile sum on the board.
        # Explain why that was (nearly) the same number every turn and what that meant
        # for the learning signal.
        if board_changed:
            step_reward = self.step_merge_score
        else:
            step_reward = self.no_move_penalty
        self.total_score += step_reward
        self.highest_tile = max(self.highest_tile, int(self.board.max()))

        new_state_ = self._get_obs()
        reward_: int = step_reward
        terminated_: bool = self.game_over
        truncated_: bool = False
        info_: dict = self._get_info(action_enum.name)

        # TODO handle all render modes in the step function
        if self.render_mode == "human":
            self._render_frame()

        return new_state_, reward_, terminated_, truncated_, info_

    def _add_new(self):
        new_val = random.choice([2, 4])
        # get a random square, if empty, add new value
        # TODO HW, this can run for arbitrary amount of time, is it a problem?
        row_ind = random.choice(range(self.num_rows))
        col_ind = random.choice(range(self.num_cols))
        current_value = self.board[row_ind, col_ind]
        if current_value == 0:
            self.board[row_ind, col_ind] = new_val
        else:
            self._add_new()

    def _slide_left(self):
        # Loop over each row, for each collapse it
        for row_ind in range(self.num_rows):
            # Loop over all cells in the column, skips left most cell, nothing to shift
            for col_ind_main in range(1, self.num_cols, 1):
                # Looping now over the previous cells to the main column cell and shift everything
                # Start with the main column index - 1 (immediate previous cell)
                # Go down to 0 (left most cell in column) by -1, so counting down
                # For each inner cell index, compare that cell with the one to the right
                # If inner cell is 0, then shift the right cell value left and make right 0
                # If inner cell is not 0, then it's blocked. We still shift if they are equal
                # If equal, double the inner cell value, set the right cell to 0, and score the merge
                # If not equal and no 0, then shifting is blocked, do nothing
                if self.board[row_ind, col_ind_main] == 0:
                    continue
                for col_ind_inner in range(
                    col_ind_main - 1, -1, -1
                ):  # needs to include zero index
                    if self.board[row_ind, col_ind_inner] == 0:
                        self.board[row_ind, col_ind_inner] = self.board[
                            row_ind, col_ind_inner + 1
                        ]
                        self.board[row_ind, col_ind_inner + 1] = 0
                    elif (
                        self.board[row_ind, col_ind_inner]
                        == self.board[row_ind, col_ind_inner + 1]
                    ):
                        self.board[row_ind, col_ind_inner] = (
                            self.board[row_ind, col_ind_inner] * 2
                        )
                        self.step_merge_score += int(self.board[row_ind, col_ind_inner])
                        self.board[row_ind, col_ind_inner + 1] = 0
                    else:
                        pass
                        continue

    def _slide_right(self):
        # Loop over each row, for each collapse it
        for row_ind in range(self.num_rows):
            # Loop over all cells in the column from the right, skips right most cell, nothing to shift
            for col_ind_main in range(
                self.num_cols - 1, -1, -1
            ):  # needs to be zero inclusive
                # Looping now over the previous cells to the main column cell and shift everything
                # Start with the main column index + 1 (immediate previous cell)
                # Go from previous cell to the last cell to the right
                # For each inner cell index, compare that cell with the one to the left
                # If inner cell is 0, then shift the left cell value right and make left 0
                # If inner cell is not 0, then it's blocked. We still shift if they are equal
                # If equal, double the inner cell value, set the left cell to 0, and score the merge
                # If not equal and no 0, then shifting is blocked, do nothing
                if self.board[row_ind, col_ind_main] == 0:
                    continue
                for col_ind_inner in range(col_ind_main + 1, self.num_cols, 1):
                    if self.board[row_ind, col_ind_inner] == 0:
                        self.board[row_ind, col_ind_inner] = self.board[
                            row_ind, col_ind_inner - 1
                        ]
                        self.board[row_ind, col_ind_inner - 1] = 0
                    elif (
                        self.board[row_ind, col_ind_inner]
                        == self.board[row_ind, col_ind_inner - 1]
                    ):
                        self.board[row_ind, col_ind_inner] = (
                            self.board[row_ind, col_ind_inner] * 2
                        )
                        self.step_merge_score += int(self.board[row_ind, col_ind_inner])
                        self.board[row_ind, col_ind_inner - 1] = 0
                    else:
                        pass
                        continue

    def _slide_up(self):
        # Loop over each column, for each collapse it
        for col_ind in range(self.num_cols):
            # Loop over all cells in the column, skips the top most cell, nothing to shift
            for row_ind_main in range(1, self.num_rows, 1):
                # Looping now over the previous cells to the main row cell and shift everything
                # Start with the main row index - 1 (immediate previous cell)
                # Go down to 0 (top most cell in column) by -1, so counting backwards
                # For each inner row index, compare that cell with the one to the below
                # If inner cell is 0, then shift the bottom cell value up and make the bottom cell 0
                # If inner cell is not 0, then it's blocked. We still shift if they are equal
                # If equal, double the inner cell value, set the bottom cell to 0, and score the merge
                # If not equal and no 0, then shifting is blocked, do nothing
                if self.board[row_ind_main, col_ind] == 0:
                    continue
                for row_ind_inner in range(
                    row_ind_main - 1, -1, -1
                ):  # inclusive of zero
                    if self.board[row_ind_inner, col_ind] == 0:
                        self.board[row_ind_inner, col_ind] = self.board[
                            row_ind_inner + 1, col_ind
                        ]
                        self.board[row_ind_inner + 1, col_ind] = 0
                    elif (
                        self.board[row_ind_inner, col_ind]
                        == self.board[row_ind_inner + 1, col_ind]
                    ):
                        self.board[row_ind_inner, col_ind] = (
                            self.board[row_ind_inner, col_ind] * 2
                        )
                        self.step_merge_score += int(self.board[row_ind_inner, col_ind])
                        self.board[row_ind_inner + 1, col_ind] = 0
                    else:
                        pass
                        continue

    def _slide_down(self):
        # Loop over each column, for each collapse it
        for col_ind in range(self.num_cols):
            # Loop over all cells in the column bottom up, skips the bottom most cell, nothing to shift
            for row_ind_main in range(self.num_rows - 1, -1, -1):  # inclusive of zero
                # Looping now over the previous cells to the main row cell and shift everything
                # Start with the main row index + 1 (immediate previous cell in the shift)
                # Go from previous cell up to max row number at the bottom
                # For each inner row index, compare that cell with the one to the above
                # If inner cell is 0, then shift the top cell value down and make the top cell 0
                # If inner cell is not 0, then it's blocked. We still shift if they are equal
                # If equal, double the inner cell value, set the top cell to 0, and score the merge
                # If not equal and no 0, then shifting is blocked, do nothing
                if self.board[row_ind_main, col_ind] == 0:
                    continue
                for row_ind_inner in range(row_ind_main + 1, self.num_rows, 1):
                    if self.board[row_ind_inner, col_ind] == 0:
                        self.board[row_ind_inner, col_ind] = self.board[
                            row_ind_inner - 1, col_ind
                        ]
                        self.board[row_ind_inner - 1, col_ind] = 0
                    elif (
                        self.board[row_ind_inner, col_ind]
                        == self.board[row_ind_inner - 1, col_ind]
                    ):
                        self.board[row_ind_inner, col_ind] = (
                            self.board[row_ind_inner, col_ind] * 2
                        )
                        self.step_merge_score += int(self.board[row_ind_inner, col_ind])
                        self.board[row_ind_inner - 1, col_ind] = 0
                    else:
                        pass
                        continue

    def _is_game_over(self):
        # The game is over only when no move can change the board: every cell is
        # filled AND no two adjacent cells are equal (nothing left to merge)
        has_empty_cell = bool(np.any(self.board == 0))
        can_merge_horizontal = bool(np.any(self.board[:, :-1] == self.board[:, 1:]))
        can_merge_vertical = bool(np.any(self.board[:-1, :] == self.board[1:, :]))
        self.game_over = not (
            has_empty_cell or can_merge_horizontal or can_merge_vertical
        )
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                board_val = self.board[i][j]
                # TODO HW: explain the conditional statement below (hint: '&' vs 'and')
                if board_val >= 2048 and (board_val & (board_val - 1) == 0):
                    pass
        if self.game_over:
            if self.window is not None:
                pygame.draw.rect(self.window, (0, 0, 0), [0, 400, 400, 100], 0, 5)
                game_over_text = self.font_.render("Game Over", 1, (240, 240, 0))
                self.window.blit(game_over_text, (10, 402))
                score_text = self.font_.render(
                    f"Total Score: {self.total_score}", 1, (240, 240, 0)
                )
                self.window.blit(score_text, (10, 450))

    def print_board(self):
        # Print for the command line
        pretty_board = PrettyTable()
        for row_ in self.board:
            pretty_board.add_row(row_)
        board_str = pretty_board.get_string(header=False, board=True)
        print(board_str)

    def render(self):
        if self.render_mode == "human":
            _ = self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            frame_dump = self._render_frame(return_rgb=True)
            return frame_dump
        elif self.render_mode == "cli":
            self.print_board()
            return None
        elif self.render_mode == "rl-headless":
            pass
            return None
        else:
            pass
            return None

    def _render_frame(self, return_rgb: bool = False):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("2048-Avalanche")
            self.font_ = pygame.font.SysFont("Verdana", 35, bold=True)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Set up color map
        cmap_ = plt.get_cmap("YlOrBr")
        norm_scale_ = mpl.colors.Normalize(vmin=0, vmax=4096, clip=True)

        # Draw initial board
        pygame.draw.rect(self.window, (200, 170, 150), [0, 0, 400, 400], 0, 5)
        pygame.draw.rect(self.window, (0, 0, 0), [0, 400, 400, 100], 0, 5)
        score_text = self.font_.render(
            f"Total Score: {self.total_score}", 1, (240, 240, 0)
        )
        self.window.blit(score_text, (10, 410))

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                value = self.board[i][j]
                if value > 16:
                    text_color = (255, 255, 255)
                else:
                    text_color = (110, 110, 110)
                value_color = cmap_(norm_scale_(value))
                pygame.draw.rect(
                    self.window, value_color, [j * 95 + 20, i * 95 + 20, 75, 75], 0, 5
                )
                value_text = self.font_.render(str(value), True, text_color)
                text_rect = value_text.get_rect(center=(j * 95 + 55, i * 95 + 55))
                self.window.blit(value_text, text_rect)
                pygame.draw.rect(
                    self.window, "black", [j * 95 + 20, i * 95 + 20, 75, 75], 2, 5
                )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if return_rgb:
            rgb_frame_dump = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
            return rgb_frame_dump

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
