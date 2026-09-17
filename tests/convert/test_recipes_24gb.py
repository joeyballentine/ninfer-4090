from __future__ import annotations

import torch

from tools.convert.methods import grouped_absmax, grouped_gptq, grouped_mse
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import Q4, Q5, Q6, Q8, qwen3_8_27b
from tools.convert.recipe import Recipe
from tools.convert.recipes.qwen3_8_27b_24gb import configure
from tools.convert.sources.logical import array_source

LAYER_PROJECTIONS = (
    ("attention/query", Q4),
    ("attention/key", Q4),
    ("attention/gate", Q5),
    ("attention/value", Q5),
    ("attention/output", Q5),
    ("gdn/query", Q4),
    ("gdn/key", Q4),
    ("gdn/value", Q5),
    ("gdn/z", Q5),
    ("gdn/output", Q5),
    ("mlp/gate", Q4),
    ("mlp/up", Q4),
    ("mlp/down", Q5),
)


def _model(layers=2):
    model = Model({"text": {"config": {"layer_types": ["full_attention"] * layers}}})

    def add(name, shape, inputs=()):
        values = torch.zeros(shape, dtype=torch.bfloat16)
        model.add(Parameter(name, shape, array_source(values, name), inputs=inputs))

    add("text/token_embedding", (256, 128))
    add("text/output_head", (256, 128), inputs=("text/final_hidden",))
    add("text/final_norm", (128,))
    for layer in range(layers):
        prefix = f"text/layers/{layer}/"
        for role, _ in LAYER_PROJECTIONS:
            add(prefix + role, (128, 128), inputs=(prefix + "mixer_input",))
        for role in ("gdn/a_projection", "gdn/b_projection"):
            add(prefix + role, (128, 128), inputs=(prefix + "mixer_input",))
        add(prefix + "input_norm", (128,))
    add("mtp/layers/0/mlp/down", (128, 128), inputs=("mtp/layers/0/mlp/product",))
    return model


def _chosen(recipe):
    return {
        name: (selections[0].format, selections[0].method)
        for name, selections in recipe.selections.items()
        if len(selections) == 1
    }


def test_official_recipe_keeps_q8_vocabulary_and_absmax() -> None:
    model = _model()
    recipe = Recipe(model)
    qwen3_8_27b(model, recipe, {})
    chosen = _chosen(recipe)
    assert chosen["text/token_embedding"] == (Q8, grouped_absmax)
    assert chosen["text/output_head"] == (Q8, grouped_absmax)
    for layer in range(2):
        for role, format in LAYER_PROJECTIONS:
            assert chosen[f"text/layers/{layer}/{role}"] == (format, grouped_absmax)


def test_24gb_recipe_moves_the_embedding_to_q6_and_searches_codes() -> None:
    model = _model()
    recipe = Recipe(model)
    configure(model, recipe, {})
    chosen = _chosen(recipe)
    assert chosen["text/token_embedding"] == (Q6, grouped_mse)
    assert chosen["text/output_head"] == (Q8, grouped_absmax)
    assert chosen["mtp/layers/0/mlp/down"] == (Q8, grouped_absmax)
    for layer in range(2):
        prefix = f"text/layers/{layer}/"
        for role, format in LAYER_PROJECTIONS:
            assert chosen[prefix + role] == (format, grouped_mse)
        for role in ("gdn/a_projection", "gdn/b_projection"):
            assert chosen[prefix + role][0] == "bf16"


def test_24gb_recipe_uses_gptq_when_calibration_is_supplied(tmp_path) -> None:
    model = _model()
    recipe = Recipe(model)
    configure(model, recipe, {}, calibration=tmp_path)
    chosen = _chosen(recipe)
    assert chosen["text/token_embedding"] == (Q6, grouped_mse)
    assert chosen["text/output_head"] == (Q8, grouped_absmax)
    for layer in range(2):
        prefix = f"text/layers/{layer}/"
        for role, format in LAYER_PROJECTIONS:
            selection = recipe.selections[prefix + role][0]
            assert (selection.format, selection.method) == (format, grouped_gptq)
            assert selection.parameters == {
                "calibration": str(tmp_path),
                "mse": True,
            }
