"""Optional SAM integration utilities for MA-SP-SAM."""

from ma_sp_sam.sam.prompt_converter import convert_prompt_package_to_sam_inputs
from ma_sp_sam.sam.sam_adapter import SAMAdapter, SAMMaskPrediction

__all__ = ["SAMAdapter", "SAMMaskPrediction", "convert_prompt_package_to_sam_inputs"]
