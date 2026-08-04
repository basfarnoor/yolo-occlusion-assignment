"""Smoke tests for the real frozen embedding model. Kept minimal and using
synthetic solid-color images (no real dataset needed) -- this just verifies
the frozen network is wired correctly and deterministic, not its actual
re-id quality (that's what the ablation studies real detections)."""
import numpy as np
import pytest
from PIL import Image

from oatm.memory.embedder import AppearanceEmbedder


@pytest.fixture(scope="module")
def embedder():
    return AppearanceEmbedder()


def test_embedding_is_l2_normalized(embedder):
    image = Image.new("RGB", (60, 60), color=(200, 30, 30))
    embedding = embedder.embed_crop(image)
    assert embedding.shape == (576,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-4


def test_embedding_is_deterministic_for_a_frozen_model(embedder):
    image = Image.new("RGB", (60, 60), color=(30, 200, 30))
    e1 = embedder.embed_crop(image)
    e2 = embedder.embed_crop(image)
    assert np.allclose(e1, e2), "a frozen model must give identical output on identical input"


def test_very_different_images_are_less_similar_than_identical_ones(embedder):
    from oatm.memory.appearance import cosine_similarity

    red = Image.new("RGB", (60, 60), color=(220, 20, 20))
    red2 = Image.new("RGB", (60, 60), color=(220, 20, 20))
    blue = Image.new("RGB", (60, 60), color=(20, 20, 220))

    e_red, e_red2, e_blue = embedder.embed_crop(red), embedder.embed_crop(red2), embedder.embed_crop(blue)
    same_color_sim = cosine_similarity(e_red, e_red2)
    diff_color_sim = cosine_similarity(e_red, e_blue)
    assert same_color_sim > diff_color_sim
