"""TTS sampling strategy: draw *n* chains from the model and vote."""

from typing import Any, Dict, List

from PIL import Image

from src.models.base import BaseVisualCoTModel
from src.voting.majority import VoteResult, majority_vote


def run_tts_sampling(
    model: BaseVisualCoTModel,
    image: Image.Image,
    query: str,
    n: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Sample *n* chains and aggregate answers by majority vote.

    Args:
        model: Any :class:`~src.models.base.BaseVisualCoTModel` instance.
        image: Input PIL image.
        query: Question string.
        n: Number of independent chains to sample.
        temperature: Stochastic sampling temperature.
        max_new_tokens: Token budget per chain.

    Returns:
        Dict with:
        - ``"chains"`` — list of raw chain dicts from the model
        - ``"vote_result"`` — :class:`~src.voting.majority.VoteResult`
        - ``"answer"`` — the voted consensus answer
    """
    chains: List[Dict[str, Any]] = model.generate(
        image, query, n=n, temperature=temperature, max_new_tokens=max_new_tokens
    )
    raw_answers = [c.get("answer", "") for c in chains]
    vote: VoteResult = majority_vote(raw_answers)
    return {"chains": chains, "vote_result": vote, "answer": vote.consensus_answer}
