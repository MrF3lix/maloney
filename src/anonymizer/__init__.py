"""Synthetic STT4SG-350 speakers made with the globi anonymizer (see anonymizer.run)."""

from anonymizer.globi import GlobiClient, GlobiError
from anonymizer.synth import LANGUAGES, plan_clips, plan_speakers

__all__ = ["LANGUAGES", "GlobiClient", "GlobiError", "plan_clips", "plan_speakers"]
