"""Qwen3.8-27B Dense for one 24 GB card, tuned for coding accuracy.

Select it with ``--recipe tools/convert/recipes/qwen3_8_27b_24gb.py``.  It
starts from the official ``qwen3_8_27b`` conversion and changes three things:

- the 248,320 x 5,120 token embedding moves from Q8 to Q6.  The embedding is a
  gather, not a matmul, so its code width costs no arithmetic; it frees
  357,780,480 bytes (1.258 GiB to 0.925 GiB) for KV pages and workspace.  The
  official ``qwen3_6_27b`` conversion already stores both vocabulary matrices
  at Q6, and the runtime carries the Q6 gather (``require_q6_metadata`` and
  ``embed_gather_q6_launch`` in ``src/ops``) plus the Q6 ``n248320_k5120``
  linear shape;
- the output head stays at Q8, where the logit margins of the full vocabulary
  are worth the bytes;
- every Q4 and Q5 Text-layer projection selects its codes with the clipping
  search instead of the plain group absmax, and with GPTQ when calibration
  Hessians are supplied.

Vision, MTP and DFlash2 assignments are exactly the official ones.
"""

from __future__ import annotations

from tools.convert.official_recipes import Q4, Q5, Q6, qwen3_8_27b

SEARCHED_FORMATS = (Q4, Q5)


def searched_projections(recipe) -> tuple[str, ...]:
    """Return the Text-layer projections the base recipe put at Q4 or Q5."""

    return tuple(
        sorted(
            name
            for name, selections in recipe.selections.items()
            if name.startswith("text/layers/")
            and len(selections) == 1
            and selections[0].format in SEARCHED_FORMATS
        )
    )


def configure(model, recipe, sources, *, calibration=None):
    qwen3_8_27b(model, recipe, sources)
    recipe.assign("text/token_embedding", format=Q6, method="grouped_mse")
    names = searched_projections(recipe)
    if calibration is None:
        recipe.assign(names, method="grouped_mse")
    else:
        recipe.assign(
            names,
            method="grouped_gptq",
            parameters={"calibration": str(calibration), "mse": True},
        )
