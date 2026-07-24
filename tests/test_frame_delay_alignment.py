import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.robot.frame_delay import FrameDelayHistory
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.datasets import RLDSBatchTransform


class RecordingActionTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, actions):
        actions = np.asarray(actions)
        self.calls.append(actions.copy())
        if actions.ndim == 1:
            return "<action>"
        return ["<action>" for _ in actions]


class DummyTokenizer:
    model_max_length = 512
    pad_token_id = 0

    def __call__(self, prompt, add_special_tokens=True):
        return SimpleNamespace(input_ids=list(range(len(prompt.split()) + 2)))


class DummyPromptBuilder:
    def __init__(self, model_family):
        self.turns = []

    def add_turn(self, role, value):
        self.turns.append((role, value))

    def get_prompt(self):
        return " ".join(value for _, value in self.turns)


class FrameDelayAlignmentTest(unittest.TestCase):
    def test_frame_history_counts_environment_steps(self):
        class MaxDelayRng:
            @staticmethod
            def randint(low, high):
                return high - 1

        history = FrameDelayHistory(max_delay_steps=5)
        for env_step in range(9):
            history.append(env_step)

        delayed_frame, delay_steps = history.sample_delayed(MaxDelayRng())

        self.assertEqual(history.current, 8)
        self.assertEqual(delay_steps, 5)
        self.assertEqual(delayed_frame, 3)

    def test_transform_uses_current_and_future_actions(self):
        action_tokenizer = RecordingActionTokenizer()
        transform = RLDSBatchTransform(
            action_tokenizer=action_tokenizer,
            base_tokenizer=DummyTokenizer(),
            image_transform=lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1),
            prompt_builder_fn=DummyPromptBuilder,
        )

        window_size = 21
        all_actions = np.arange((window_size + NUM_ACTIONS_CHUNK - 1) * ACTION_DIM, dtype=np.float32).reshape(
            window_size + NUM_ACTIONS_CHUNK - 1, ACTION_DIM
        )
        primary_images = np.zeros((window_size, 4, 4, 3), dtype=np.uint8)
        primary_images[-1].fill(255)
        rlds_batch = {
            "dataset_name": "libero_goal_no_noops",
            "action": all_actions,
            "observation": {"image_primary": primary_images},
            "task": {"language_instruction": b"open the drawer"},
        }

        result = transform(rlds_batch)

        np.testing.assert_array_equal(result["actions"], all_actions[-NUM_ACTIONS_CHUNK:])
        np.testing.assert_array_equal(action_tokenizer.calls[0], all_actions[-(NUM_ACTIONS_CHUNK - 1) :])
        np.testing.assert_array_equal(action_tokenizer.calls[1], all_actions[-NUM_ACTIONS_CHUNK])
        self.assertTrue(torch.all(result["pixel_values"] == 255))

    def test_collator_rejects_historical_actions(self):
        collator = PaddedCollatorForActionPrediction(model_max_length=32, pad_token_id=0)
        instance = {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
            "pixel_values": torch.zeros(6, 4, 4),
            "actions": np.zeros((28, ACTION_DIM), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "current action chunk"):
            collator([instance])


if __name__ == "__main__":
    unittest.main()
