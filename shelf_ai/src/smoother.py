"""
smoother.py
-----------
Temporal detection smoother for real-time shelf monitoring.

When analysing a live video feed, individual frames can produce noisy
detection counts (missed detections, false positives).  The
DetectionSmoother maintains a rolling window of the last *N* frames and
returns time-averaged counts, dramatically reducing per-frame variance
without introducing significant lag.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict


class DetectionSmoother:
    """
    Rolling-window mean smoother for per-product detection counts.

    Each call to :meth:`update` records the current frame's counts and
    returns the rounded mean across the last ``window`` frames.  Products
    not seen in the current frame are treated as zero for that frame (so
    the mean naturally falls when a product disappears from view).

    Parameters
    ----------
    window : int
        Number of frames to average over (default: 5).  Must be ≥ 1.

    Example
    -------
    >>> smoother = DetectionSmoother(window=5)
    >>> smoothed = smoother.update({"maggi": 3, "lays": 1})
    >>> smoothed = smoother.update({"maggi": 4, "lays": 2})
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, counts: Dict[str, int]) -> Dict[str, int]:
        """
        Push a new per-product count snapshot and return smoothed counts.

        Products not present in *counts* are treated as 0 for this frame.
        Products that have never been seen are omitted from the output.

        Parameters
        ----------
        counts : dict
            Mapping of ``product_name -> detected_count`` for one frame.

        Returns
        -------
        dict
            Smoothed mapping of ``product_name -> smoothed_count``.
        """
        all_known = set(self._buffers.keys()) | set(counts.keys())
        for product in all_known:
            self._buffers[product].append(counts.get(product, 0))
        return {
            product: round(sum(buf) / len(buf))
            for product, buf in self._buffers.items()
            if buf
        }

    def reset(self) -> None:
        """Clear all buffered frame history."""
        self._buffers.clear()

    @property
    def window(self) -> int:
        """The configured smoothing window size."""
        return self._window
