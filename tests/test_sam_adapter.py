import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models.sam_adapter import CandidateMask, SAMAdapter
from ma_sp_sam.prompts import PromptSynthesizer


class FakeImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.img_size = 16

    def forward(self, images):
        batch = images.shape[0]
        return self.weight * torch.ones(batch, 4, 4, 4, device=images.device)


class FakePromptEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.input_image_size = (16, 16)
        self.image_embedding_size = (4, 4)
        self.mask_input_size = (16, 16)
        self.last_points = None
        self.last_boxes = None
        self.last_masks = None

    def forward(self, points, boxes, masks):
        coords, labels = points
        self.last_points = (coords.detach().clone(), labels.detach().clone())
        self.last_boxes = boxes.detach().clone()
        self.last_masks = masks.detach().clone()
        batch = boxes.shape[0]
        sparse = torch.zeros(batch, coords.shape[1] + 2, 4, device=boxes.device)
        dense = torch.zeros(batch, 4, 4, 4, device=boxes.device)
        return sparse, dense

    def get_dense_pe(self):
        return torch.zeros(1, 4, 4, 4, device=self.weight.device)


class FakeMaskDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.last_batch = None
        self.last_multimask_output = None

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        self.last_batch = image_embeddings.shape[0]
        self.last_multimask_output = multimask_output
        channels = 3 if multimask_output else 1
        low_res = self.weight * torch.ones(image_embeddings.shape[0], channels, 8, 8, device=image_embeddings.device)
        iou = torch.arange(image_embeddings.shape[0] * channels, dtype=torch.float32, device=image_embeddings.device).reshape(
            image_embeddings.shape[0], channels
        )
        return low_res, iou


class FakeSam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeImageEncoder()
        self.prompt_encoder = FakePromptEncoder()
        self.mask_decoder = FakeMaskDecoder()

    def preprocess(self, image):
        return image


def _prompt_result():
    proposals = np.zeros((8, 8), dtype=np.uint16)
    proposals[1:4, 1:4] = 7
    proposals[4:7, 4:7] = 9
    center = np.zeros((8, 8), dtype=np.float32)
    center[2, 2] = 1
    center[5, 5] = 0.9
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[proposals > 0] = 2
    boundary = np.zeros((2, 8, 8), dtype=np.float32)
    return PromptSynthesizer(max_negative_points=1).synthesize(
        center_heatmap=center,
        semantic_logits=semantic,
        boundary_maps=boundary,
        instance_proposals=proposals,
    )


def test_sam_adapter_configures_trainable_modules():
    sam = FakeSam()

    adapter = SAMAdapter(sam, freeze_image_encoder=True, train_mask_decoder=True)

    assert not any(param.requires_grad for param in adapter.sam.image_encoder.parameters())
    assert not any(param.requires_grad for param in adapter.sam.prompt_encoder.parameters())
    assert all(param.requires_grad for param in adapter.sam.mask_decoder.parameters())
    assert not adapter.sam.image_encoder.training
    assert adapter.sam.mask_decoder.training


def test_sam_adapter_decodes_prompt_synthesizer_packages_to_candidate_masks():
    sam = FakeSam()
    adapter = SAMAdapter(sam)

    output = adapter(
        image_embeddings=torch.ones(1, 4, 4, 4),
        prompts=_prompt_result(),
        prompt_image_size=(8, 8),
        output_size=(16, 16),
        multimask_output=True,
    )

    assert [candidate.instance_id for candidate in output.candidates] == [7, 9]
    assert all(isinstance(candidate, CandidateMask) for candidate in output.candidates)
    assert output.masks.shape == (2, 3, 16, 16)
    assert output.low_res_logits.shape == (2, 3, 8, 8)
    assert output.iou_predictions.shape == (2, 3)
    assert sam.mask_decoder.last_batch == 2
    assert sam.mask_decoder.last_multimask_output is True


def test_sam_adapter_scales_prompts_and_resizes_coarse_masks():
    sam = FakeSam()
    adapter = SAMAdapter(sam)

    adapter(
        image_embeddings=torch.ones(1, 4, 4, 4),
        prompts=_prompt_result(),
        prompt_image_size=(8, 8),
        output_size=None,
    )

    coords, labels = sam.prompt_encoder.last_points
    assert tuple(coords[0, 0].tolist()) == (4.0, 4.0)
    assert labels[0, 0].item() == 1
    assert tuple(sam.prompt_encoder.last_boxes[0].tolist()) == (2.0, 2.0, 6.0, 6.0)
    assert sam.prompt_encoder.last_masks.shape == (2, 1, 16, 16)


def test_from_pretrained_loads_standard_sam_registry(monkeypatch):
    fake_module = types.ModuleType("segment_anything")
    seen = {}

    def build_vit_b(checkpoint=None):
        seen["checkpoint"] = checkpoint
        return FakeSam()

    fake_module.sam_model_registry = {"vit_b": build_vit_b}
    monkeypatch.setitem(__import__("sys").modules, "segment_anything", fake_module)

    adapter = SAMAdapter.from_pretrained(
        backend="sam",
        model_type="vit_b",
        checkpoint_path="/tmp/fake_sam_vit_b.pth",
        source_root="/tmp/fake_segment_anything",
        device="cpu",
    )

    assert isinstance(adapter, SAMAdapter)
    assert seen["checkpoint"] == "/tmp/fake_sam_vit_b.pth"
