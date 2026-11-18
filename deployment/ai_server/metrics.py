"""Per-attempt timing metrics using the server monotonic clock."""
from collections import deque
import time


class AttemptMetrics:
    def __init__(self):
        self.started = time.monotonic()
        self.first_received = None
        self.last_received = None
        self.received = deque(maxlen=16)
        self.frame_count = 0
        self.pass_seconds = None

    def receive_frame(self):
        now = time.monotonic()
        if self.first_received is None:
            self.first_received = now
        self.last_received = now
        self.received.append(now)
        self.frame_count += 1

    def snapshot(self):
        now = time.monotonic()
        recent_span = self.received[-1] - self.received[0] if len(self.received) > 1 else 0
        total_span = self.last_received - self.first_received if self.frame_count > 1 else 0
        return dict(
            frame_count=self.frame_count,
            elapsed_seconds=round(now - self.started, 3),
            received_fps=round((len(self.received) - 1) / recent_span, 2) if recent_span > 0 else None,
            average_received_fps=round((self.frame_count - 1) / total_span, 2) if total_span > 0 else None,
            pass_seconds=self.pass_seconds,
        )
