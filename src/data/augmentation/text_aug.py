"""Rule-based text / question paraphrasing for test-time scaling."""

import random
from typing import List, Optional, Tuple


class TextAugmentor:
    """Generate meaning-preserving paraphrases of questions.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paraphrase_question(self, question: str) -> str:
        """Apply a random rule-based transformation to *question*.

        Args:
            question: Original question text.

        Returns:
            A rephrased version of the question.
        """
        strategy = random.choice(
            [
                self._add_politeness,
                self._rephrase_what_is,
                self._add_specificity,
                self._formal_informal,
                self._reorder_clauses,
            ]
        )
        return strategy(question)

    def generate_text_variants(
        self, question: str, num_paraphrases: int = 3
    ) -> List[Tuple[str, str]]:
        """Generate multiple paraphrases of *question*.

        Args:
            question: Original question text.
            num_paraphrases: How many paraphrases to produce.

        Returns:
            List of ``(paraphrased_question, variant_id)`` tuples, starting
            with the original as ``"original"``.
        """
        variants: List[Tuple[str, str]] = [(question, "original")]
        for i in range(num_paraphrases):
            variants.append((self.paraphrase_question(question), f"paraphrase_{i + 1}"))
        return variants

    # ------------------------------------------------------------------
    # Private transformation strategies
    # ------------------------------------------------------------------

    def _add_politeness(self, question: str) -> str:
        q = question.rstrip("?")
        prefix = random.choice(
            [
                "Could you please tell me ",
                "I would like to know ",
                "Can you identify ",
            ]
        )
        return prefix + q.lower() + "?"

    def _rephrase_what_is(self, question: str) -> str:
        replacements = {
            "What is": "Can you identify what",
            "What are": "Can you identify what",
            "Which": "Please select which",
            "How many": "What is the count of",
        }
        for old, new in replacements.items():
            if question.startswith(old):
                return question.replace(old, new, 1)
        return question

    def _add_specificity(self, question: str) -> str:
        q = question.rstrip("?")
        addition = random.choice(
            [
                " based on the image",
                " according to what you see",
                " from the given image",
                " as shown in the picture",
            ]
        )
        return q + addition + "?"

    def _formal_informal(self, question: str) -> str:
        mappings = {
            "identify": "spot",
            "determine": "figure out",
            "select": "pick",
            "indicate": "show",
        }
        lower = question.lower()
        for formal, informal in mappings.items():
            if formal in lower:
                return lower.replace(formal, informal).capitalize()
        return question

    def _reorder_clauses(self, question: str) -> str:
        if " in the image" in question:
            return "In the image, " + question.replace(" in the image", "")
        if " from the image" in question:
            return "From the image, " + question.replace(" from the image", "")
        return question
