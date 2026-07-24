from collections import deque


class FrameDelayHistory:
    """Bounded frame history where each entry represents one environment step."""

    def __init__(self, max_delay_steps: int):
        if max_delay_steps < 1:
            raise ValueError("max_delay_steps must be at least 1")
        self.max_delay_steps = max_delay_steps
        self._frames = deque(maxlen=max_delay_steps + 1)

    def append(self, frame) -> None:
        self._frames.append(frame)

    @property
    def current(self):
        if not self._frames:
            raise RuntimeError("Cannot read current frame from an empty history")
        return self._frames[-1]

    def sample_delayed(self, rng):
        if len(self._frames) < 2:
            return None, 0

        max_available_delay = min(self.max_delay_steps, len(self._frames) - 1)
        delay_steps = int(rng.randint(1, max_available_delay + 1))
        return self._frames[-1 - delay_steps], delay_steps
