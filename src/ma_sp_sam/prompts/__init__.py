"""Prompt synthesis utilities for MA-SP-SAM."""

from ma_sp_sam.prompts.prompt_synthesizer import (
    BoxPrompt,
    CoarseMaskPrompt,
    InstanceProposal,
    PointPrompt,
    PromptPackage,
    PromptSynthesisResult,
    PromptSynthesizer,
    RingPrompt,
    synthesize_prompts,
)

__all__ = [
    "BoxPrompt",
    "CoarseMaskPrompt",
    "InstanceProposal",
    "PointPrompt",
    "PromptPackage",
    "PromptSynthesisResult",
    "PromptSynthesizer",
    "RingPrompt",
    "synthesize_prompts",
]
