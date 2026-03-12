"""Answer normalisation utilities."""

import re


def normalize_answer(raw_answer: str) -> str:
    """Normalise a raw model answer to the canonical A/B/C/D format.

    Handles a wide variety of surface forms including:
    - ``"A"``, ``"(B)"``, ``"[C]"``, ``"{D}"``
    - ``"option A"``, ``"Option B:"``, ``"answer: C"``
    - ``"The answer is D"``
    - Full text starting with the answer letter

    Args:
        raw_answer: Raw answer string from the model.

    Returns:
        Normalised answer (A/B/C/D) or the original string if no match.
    """
    if not raw_answer:
        return ""

    answer = str(raw_answer).strip().upper()

    # Direct single letter
    if len(answer) == 1 and answer in "ABCDEFGH":
        return answer

    # Pattern 1: letter inside brackets / parens  e.g. (A), [B], {C}, <D>
    m = re.search(r"[\(\[\{<]\s*([A-H])\s*[\)\]\}>]", answer)
    if m:
        return m.group(1)

    # Pattern 2: keyword-prefixed  e.g. "Option A", "Answer: B"
    m = re.search(r"(?:OPTION|ANSWER|CHOICE|SELECT)\s*:?\s*([A-H])", answer)
    if m:
        return m.group(1)

    # Pattern 3: "The answer is A"
    m = re.search(r"(?:IS|ARE)\s+([A-H])(?:\s|$|\.)", answer)
    if m:
        return m.group(1)

    # Pattern 4: letter followed by punctuation or end
    m = re.search(r"^([A-H])[\s\.\,\:\;\)\]]", answer)
    if m:
        return m.group(1)

    # Pattern 5: letter at start of string
    m = re.search(r"^([A-H])", answer)
    if m:
        return m.group(1)

    # Pattern 6: first isolated letter A-H
    matches = re.findall(r"\b([A-H])\b", answer)
    if matches:
        return matches[0]

    return answer
