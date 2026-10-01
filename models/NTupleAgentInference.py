import numpy as np

import fast_engine
import ntuple
from expectimax import ExpectimaxSearch


class NTupleAgentInference:

    def __init__(self, model_file_path: str, search_depth: int = 1,
                 chance_cap: int = 6, time_budget_ms=None,
                 deepen_empty_threshold: int = 0, deepen_extra: int = 0):
        weights, shapes, self.games_trained, thresholds = (
            ntuple.load_checkpoint_full(model_file_path)
        )
        self.shifts, self.bases, n_weights = ntuple.build_placements(shapes)
        # single-stage checkpoints hold one table set; staged checkpoints hold
        # one per stage, routed by the board's largest tile
        if weights.ndim == 1:
            weights = weights[None, :]
        self.stage_thresholds = thresholds if thresholds is not None else ()
        if weights.shape[0] != len(self.stage_thresholds) + 1:
            raise ValueError(
                f"checkpoint has {weights.shape[0]} stage table(s) but "
                f"{len(self.stage_thresholds)} threshold(s)"
            )
        if weights.shape[1] != n_weights:
            raise ValueError(
                f"checkpoint weight count {weights.shape[1]} does not match "
                f"its shape set (expected {n_weights})"
            )
        self.stage_weights = [np.ascontiguousarray(w) for w in weights]
        self.searches = None
        if search_depth > 1 or deepen_extra > 0:
            self.searches = [
                ExpectimaxSearch(
                    stage_table, self.shifts, self.bases, depth=search_depth,
                    chance_cap=chance_cap, time_budget_ms=time_budget_ms,
                    deepen_empty_threshold=deepen_empty_threshold,
                    deepen_extra=deepen_extra,
                )
                for stage_table in self.stage_weights
            ]

    def _stage(self, obs_raw) -> int:
        max_value = int(np.max(obs_raw))
        max_exp = max_value.bit_length() - 1 if max_value > 0 else 0
        return ntuple.stage_for_exponent(max_exp, self.stage_thresholds)

    def get_action(self, obs_raw: np.ndarray) -> int:
        board = np.uint64(fast_engine.board_from_array(obs_raw))
        stage = self._stage(obs_raw)
        if self.searches is not None:
            action = self.searches[stage].best_action(board)
        else:
            action = int(
                ntuple.choose_action(
                    board, self.stage_weights[stage], self.shifts, self.bases
                )
            )
        # -1 means no move can change the board; the env is about to end the
        # game, so any action is equivalent
        return action if action >= 0 else 0
