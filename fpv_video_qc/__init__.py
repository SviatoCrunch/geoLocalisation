"""fpv_video_qc — detect corrupted (RF-breakup) segments in analog FPV video.

Per-frame CPU metrics (cv2/numpy) score each frame for the analog-FPV failure modes — horizontal
line breakup / dropped-or-torn rows, sudden temporal discontinuity (sync loss / frame drop), and
impulse chroma noise — flag bad frames, and group them into segments to drop. Output = a JSON
timeline + a visual montage (worst vs clean frames, score-labelled) so thresholds can be tuned by eye
before anything is cut. See [[project_fpv_video_restoration]].
"""
from .core import (bad_segments, frame_score, line_noise_from_gray, FRAME_KEYS)

__all__ = ["bad_segments", "frame_score", "line_noise_from_gray", "FRAME_KEYS"]
