"""TTS scaling strategy: run inference over augmented views and vote."""

from typing import Any, Dict, List, Optional

from PIL import Image

from src.data.augmentation import AugmentationConfig, generate_views
from src.models.base import BaseVisualCoTModel
from src.voting.majority import VoteResult, majority_vote


def run_tts_scaling(
    model: BaseVisualCoTModel,
    image: Image.Image,
    query: str,
    options: Optional[Dict[str, str]] = None,
    config: Optional[AugmentationConfig] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run inference over augmented image/text views and vote.

    Each view produced by :func:`~src.data.augmentation.generate_views` is
    passed to the model independently; the resulting answers are aggregated
    by majority vote.

    Args:
        model: Any :class:`~src.models.base.BaseVisualCoTModel` instance.
        image: Original PIL image.
        query: Original question string.
        options: Answer options dict (optional, used only for view generation).
        config: Augmentation configuration; uses defaults if ``None``.
        temperature: Sampling temperature per view (0 = greedy).
        max_new_tokens: Token budget per view.

    Returns:
        Dict with:
        - ``"views"`` — the augmented view list
        - ``"chains"`` — one chain dict per view
        - ``"vote_result"`` — :class:`~src.voting.majority.VoteResult`
        - ``"answer"`` — the voted consensus answer
    """
    if options is None:
        options = {}

    views = generate_views(image, query, options, config)
    chains: List[Dict[str, Any]] = []
    for view in views:
        chain = model.predict(
            view["image"],
            view["question"],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        chains.append(chain)

    raw_answers = [c.get("answer", "") for c in chains]
    vote: VoteResult = majority_vote(raw_answers)
    return {"views": views, "chains": chains, "vote_result": vote, "answer": vote.consensus_answer}
