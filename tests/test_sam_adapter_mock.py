import sys
import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.prompts import PromptSynthesizer


class FakeSam:
    def __init__(self):
        self.device = None

    def to(self, device=None):
        self.device = device
        return self


class FakePredictor:
    def __init__(self, sam):
        self.sam = sam
        self.image = None
        self.calls = []

    def set_image(self, image):
        self.image = image

    def predict(self, *, point_coords, point_labels, box, mask_input, multimask_output):
        self.calls.append(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "box": box,
                "mask_input": mask_input,
                "multimask_output": multimask_output,
            }
        )
        height, width = self.image.shape[:2]
        count = 3 if multimask_output else 1
        masks = np.zeros((count, height, width), dtype=bool)
        masks[:, 1:4, 1:4] = True
        scores = np.asarray([0.1, 0.8, 0.4][:count], dtype=np.float32)
        logits = np.ones((count, 4, 4), dtype=np.float32)
        return masks, scores, logits


def _install_fake_segment_anything(monkeypatch):
    fake_module = types.ModuleType("segment_anything")
    fake_module.sam_model_registry = {"vit_b": lambda checkpoint=None: FakeSam()}
    fake_module.SamPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "segment_anything", fake_module)


def _packages():
    proposals = np.zeros((8, 8), dtype=np.uint16)
    proposals[1:4, 1:4] = 3
    proposals[4:7, 4:7] = 5
    semantic = np.zeros_like(proposals, dtype=np.uint8)
    semantic[proposals > 0] = 2
    center = np.zeros_like(proposals, dtype=np.float32)
    center[2, 2] = 1.0
    center[5, 5] = 0.9
    boundary = np.zeros((2, 8, 8), dtype=np.float32)
    return PromptSynthesizer(max_negative_points=1).synthesize(
        center_heatmap=center,
        semantic_logits=semantic,
        boundary_maps=boundary,
        instance_proposals=proposals,
    ).packages


def test_sam_adapter_mock_predicts_from_packages_and_converts_grayscale_to_rgb(monkeypatch):
    _install_fake_segment_anything(monkeypatch)
    from ma_sp_sam.sam.sam_adapter import SAMAdapter, SAMMaskPrediction

    adapter = SAMAdapter(checkpoint="/tmp/fake_sam.pth", model_type="vit_b", device="cpu")
    image = torch.linspace(0, 1, 64, dtype=torch.float32).reshape(1, 8, 8)

    predictions = adapter.predict_from_packages(image, _packages(), multimask_output=False)

    assert adapter.is_available()
    assert [prediction.instance_id for prediction in predictions] == [3, 5]
    assert all(isinstance(prediction, SAMMaskPrediction) for prediction in predictions)
    assert predictions[0].masks.shape == (1, 8, 8)
    assert predictions[0].scores.shape == (1,)
    assert predictions[0].best_index == 0
    assert adapter.predictor.image.shape == (8, 8, 3)
    assert adapter.predictor.image.dtype == np.uint8
    assert adapter.predictor.calls[0]["mask_input"] is None
    assert adapter.predictor.calls[0]["point_labels"][0] == 1
    assert adapter.predictor.calls[0]["box"].tolist() == [1.0, 1.0, 3.0, 3.0]


def test_sam_adapter_can_enable_low_res_mask_input_and_tracks_best_index(monkeypatch):
    _install_fake_segment_anything(monkeypatch)
    from ma_sp_sam.sam.sam_adapter import SAMAdapter, best_mask

    adapter = SAMAdapter(checkpoint="/tmp/fake_sam.pth", model_type="vit_b", device="cpu")
    image = np.zeros((8, 8), dtype=np.uint8)

    prediction = adapter.predict_from_package(
        image,
        _packages()[0],
        multimask_output=True,
        use_mask_input=True,
    )

    assert adapter.predictor.calls[0]["mask_input"].shape == (1, 256, 256)
    assert prediction.best_index == 1
    assert best_mask(prediction).shape == (8, 8)
    assert prediction.prompt_metadata["coarse_mask_shape"] == (8, 8)


def test_sam_adapter_missing_segment_anything_error_is_clear(monkeypatch):
    from ma_sp_sam.sam import sam_adapter

    def missing_import(name):
        raise ImportError("not installed")

    monkeypatch.setattr(sam_adapter.importlib, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="segment-anything is not installed. Install it to use SAMAdapter."):
        sam_adapter.SAMAdapter(checkpoint="/tmp/fake_sam.pth")
