"""Tests for src.models and src.methods (using mocked model)."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models.base import BaseVisualCoTModel
from src.methods.baseline import run_baseline
from src.methods.tts.sampling import run_tts_sampling
from src.methods.tts.scaling import run_tts_scaling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_IMAGE = Image.new("RGB", (64, 64), color=(10, 20, 30))
QUERY = "What is shown in the image?"
OPTIONS = {"A": "Cat", "B": "Dog", "C": "Fish", "D": "Bird"}


def _make_chain(answer: str = "A") -> Dict[str, Any]:
    return {"bbox_raw": "[0.1, 0.1, 0.9, 0.9]", "coords": [0.1, 0.1, 0.9, 0.9], "answer": answer}


class _MockModel(BaseVisualCoTModel):
    """Deterministic mock: always returns the same chain."""

    def __init__(self, answer: str = "A"):
        self._answer = answer
        self.call_count = 0

    def generate(self, image, query, *, n=1, temperature=0.0, max_new_tokens=512, **kw):
        self.call_count += n
        return [_make_chain(self._answer) for _ in range(n)]


# ---------------------------------------------------------------------------
# TestBaseVisualCoTModel
# ---------------------------------------------------------------------------


class TestBaseVisualCoTModel:
    def test_predict_calls_generate_with_n_1(self):
        model = _MockModel("B")
        result = model.predict(TINY_IMAGE, QUERY)
        assert model.call_count == 1
        assert result["answer"] == "B"

    def test_predict_returns_single_dict(self):
        model = _MockModel("C")
        result = model.predict(TINY_IMAGE, QUERY)
        assert isinstance(result, dict)

    def test_generate_returns_list_of_n_chains(self):
        model = _MockModel("A")
        chains = model.generate(TINY_IMAGE, QUERY, n=5)
        assert len(chains) == 5


# ---------------------------------------------------------------------------
# TestRunBaseline
# ---------------------------------------------------------------------------


class TestRunBaseline:
    def test_run_baseline_returns_chain_dict(self):
        model = _MockModel("D")
        result = run_baseline(model, TINY_IMAGE, QUERY)
        assert "answer" in result

    def test_run_baseline_calls_model_once(self):
        model = _MockModel("A")
        run_baseline(model, TINY_IMAGE, QUERY)
        assert model.call_count == 1


# ---------------------------------------------------------------------------
# TestTTSSampling
# ---------------------------------------------------------------------------


class TestTTSSampling:
    @pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
    def test_tts_method_calls_generate_exactly_n_times(self, n):
        model = _MockModel("A")
        run_tts_sampling(model, TINY_IMAGE, QUERY, n=n)
        assert model.call_count == n

    def test_tts_sampling_returns_answer_key(self):
        model = _MockModel("B")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=3)
        assert "answer" in result

    def test_tts_sampling_answer_is_from_vote(self):
        model = _MockModel("C")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=5)
        assert result["answer"] == "C"

    def test_tts_sampling_returns_chains_list(self):
        model = _MockModel("A")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=4)
        assert len(result["chains"]) == 4

    def test_tts_sampling_vote_result_present(self):
        from src.voting.majority import VoteResult

        model = _MockModel("A")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=3)
        assert isinstance(result["vote_result"], VoteResult)


# ---------------------------------------------------------------------------
# TestTTSScaling
# ---------------------------------------------------------------------------


class TestTTSScaling:
    def test_tts_scaling_returns_answer_key(self):
        model = _MockModel("A")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert "answer" in result

    def test_tts_scaling_returns_views_and_chains(self):
        model = _MockModel("B")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert "views" in result
        assert "chains" in result
        assert len(result["views"]) == len(result["chains"])

    def test_tts_scaling_answer_matches_majority(self):
        model = _MockModel("D")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert result["answer"] == "D"


# ---------------------------------------------------------------------------
# TestDeepEyesV2Model
# ---------------------------------------------------------------------------


class TestDeepEyesV2Model:
    """Tests for DeepEyesV2Model — all model weights are mocked."""

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _make_model() -> "DeepEyesV2Model":  # type: ignore[name-defined]
        """Instantiate DeepEyesV2Model with stubbed HF components."""
        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = DeepEyesV2Model.__new__(DeepEyesV2Model)
        model.model_id = "honglyhly/DeepEyesV2_7B_1031"
        model.max_turns = 10
        model._model = MagicMock()
        model._processor = MagicMock()
        model._processor.apply_chat_template.return_value = "fake prompt"
        model._processor.return_value = {
            "input_ids": MagicMock(shape=(1, 10)),
            "attention_mask": MagicMock(),
        }
        model._processor.batch_decode.return_value = ["<answer>Paris</answer>"]
        model._model.generate.return_value = MagicMock()
        model._model.device = "cpu"
        return model

    # ---- BaseVisualCoTModel contract ---------------------------------------

    def test_generate_returns_list_of_n_dicts(self):
        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "Rome", "cot_steps": [], "tool_results": []}
        with patch.object(model, "_run_chain", return_value=stub):
            chains = model.generate(TINY_IMAGE, QUERY, n=3)
        assert len(chains) == 3

    def test_generate_chain_has_required_keys(self):
        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "Berlin", "cot_steps": ["s"], "tool_results": []}
        with patch.object(model, "_run_chain", return_value=stub):
            chain = model.generate(TINY_IMAGE, QUERY, n=1)[0]
        for key in ("bbox_raw", "coords", "answer", "cot_steps", "tool_results"):
            assert key in chain

    def test_generate_bbox_raw_is_none_and_coords_empty(self):
        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "Paris", "cot_steps": [], "tool_results": []}
        with patch.object(model, "_run_chain", return_value=stub):
            chain = model.generate(TINY_IMAGE, QUERY, n=1)[0]
        assert chain["bbox_raw"] is None
        assert chain["coords"] == []

    @pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
    def test_generate_calls_run_chain_exactly_n_times(self, n):
        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "X", "cot_steps": [], "tool_results": []}
        with patch.object(model, "_run_chain", return_value=stub) as mock_chain:
            model.generate(TINY_IMAGE, QUERY, n=n)
        assert mock_chain.call_count == n

    # ---- answer parsing ----------------------------------------------------

    def test_parse_answer_extracts_text_between_tags(self):
        from src.models.deepeyes_v2 import _parse_answer

        assert _parse_answer("<answer>London</answer>") == "London"

    def test_parse_answer_returns_none_when_no_tag(self):
        from src.models.deepeyes_v2 import _parse_answer

        assert _parse_answer("I think the answer is Paris.") is None

    def test_parse_answer_strips_whitespace_inside_tags(self):
        from src.models.deepeyes_v2 import _parse_answer

        assert _parse_answer("<answer>  Tokyo  </answer>") == "Tokyo"

    def test_parse_answer_handles_multiline_content(self):
        from src.models.deepeyes_v2 import _parse_answer

        assert _parse_answer("<answer>\nNew York\n</answer>") == "New York"

    # ---- code block extraction ---------------------------------------------

    def test_extract_code_block_finds_code_tags(self):
        from src.models.deepeyes_v2 import _extract_code_block

        text = "Some reasoning.\n<code>print('hi')</code>\nMore text."
        assert _extract_code_block(text) == "print('hi')"

    def test_extract_code_block_returns_none_when_absent(self):
        from src.models.deepeyes_v2 import _extract_code_block

        assert _extract_code_block("No code here.") is None

    def test_extract_code_block_strips_python_markdown_wrapper(self):
        """Model generates '''python ... ''' inside <code> tags; we must unwrap it."""
        from src.models.deepeyes_v2 import _extract_code_block

        text = "<code>\n'''python\nprint(image_1.size)\n'''\n</code>"
        result = _extract_code_block(text)
        assert result == "print(image_1.size)"

    def test_extract_code_block_strips_backtick_fence(self):
        """Also handle ```python ... ``` variant inside <code> tags."""
        from src.models.deepeyes_v2 import _extract_code_block

        text = "<code>\n```python\nprint(42)\n```\n</code>"
        result = _extract_code_block(text)
        assert result == "print(42)"

    def test_extract_code_block_detects_top_level_backtick_fence(self):
        """Model may emit bare ```python fences without any <code> wrapper."""
        from src.models.deepeyes_v2 import _extract_code_block

        text = "Some reasoning.\n```python\nprint(image_1.size)\n```\nMore text."
        result = _extract_code_block(text)
        assert result == "print(image_1.size)"

    def test_extract_code_block_prefers_code_tag_over_bare_fence(self):
        """<code> tags should take priority when both formats appear."""
        from src.models.deepeyes_v2 import _extract_code_block

        text = "```python\nfoo()\n```\n<code>bar()</code>"
        result = _extract_code_block(text)
        assert result == "bar()"

    # ---- code executor -----------------------------------------------------

    def test_execute_code_returns_stdout(self):
        import numpy as np

        from src.models.deepeyes_v2 import _execute_code

        ns: dict = {"image_1": np.zeros((64, 64, 3), dtype=np.uint8), "np": np}
        result = _execute_code("print('hello')", ns)
        assert result == "hello"

    def test_execute_code_uses_image_namespace(self):
        import numpy as np

        from src.models.deepeyes_v2 import _execute_code

        img = np.ones((10, 10, 3), dtype=np.uint8) * 42
        ns: dict = {"image_1": img, "np": np}
        result = _execute_code("print(image_1.shape)", ns)
        assert "(10, 10, 3)" in result

    def test_execute_code_captures_exception_message(self):
        import numpy as np

        from src.models.deepeyes_v2 import _execute_code

        ns: dict = {"image_1": np.zeros((4, 4, 3), dtype=np.uint8), "np": np}
        result = _execute_code("raise ValueError('boom')", ns)
        assert "ValueError" in result
        assert "boom" in result

    def test_execute_code_empty_output_returns_empty_string(self):
        import numpy as np

        from src.models.deepeyes_v2 import _execute_code

        ns: dict = {"image_1": np.zeros((4, 4, 3), dtype=np.uint8), "np": np}
        result = _execute_code("x = 1 + 1", ns)
        assert result == ""

    def test_execute_code_plt_show_does_not_block(self):
        """plt.show() inside the sandbox must never open a GUI window."""
        matplotlib = pytest.importorskip("matplotlib")
        from src.models.deepeyes_v2 import _execute_code

        code = (
            "import matplotlib\n"
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1, 2, 3])\n"
            "plt.show()\n"
            "print('after_show')"
        )
        result = _execute_code(code, {})
        assert result == "after_show"

    def test_execute_code_matplotlib_uses_non_interactive_backend(self):
        """After executing sandbox code the matplotlib backend must be non-GUI."""
        pytest.importorskip("matplotlib")
        from src.models.deepeyes_v2 import _execute_code

        result = _execute_code(
            "import matplotlib; print(matplotlib.get_backend())", {}
        )
        assert result.lower() in {"agg", "pdf", "svg", "ps", "cairo"}, (
            f"Expected a non-interactive backend, got {result!r}"
        )

    # ---- PIL image in namespace --------------------------------------------

    def test_run_chain_image_1_is_pil_image(self):
        """image_1 in the exec namespace must be a PIL Image (not NumPy array)."""
        from unittest.mock import patch

        from PIL import Image

        from src.models.deepeyes_v2 import DeepEyesV2Model, _execute_code

        model = self._make_model()
        captured_ns: dict = {}

        def fake_execute(code: str, namespace: dict) -> str:
            captured_ns.update(namespace)
            return ""

        with patch.object(model, "_call_model", side_effect=[
            "<code>\n'''python\nprint(image_1.size)\n'''\n</code>",
            "<answer>done</answer>",
        ]), patch("src.models.deepeyes_v2._execute_code", side_effect=fake_execute):
            model._run_chain(TINY_IMAGE, QUERY, temperature=0.0, max_new_tokens=512)

        assert "image_1" in captured_ns
        assert isinstance(captured_ns["image_1"], Image.Image), (
            "image_1 must be a PIL Image; model code uses PIL methods like .crop(), .resize()"
        )

    def test_run_chain_tool_result_uses_code_execution_format(self):
        """Tool results must match RETURN_CODE_USER_PROMPT from the paper (Appendix A.8)."""
        from unittest.mock import patch

        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = self._make_model()
        injected_messages: list = []

        turn_responses = iter([
            "<code>\n'''python\nprint('hello')\n'''\n</code>",
            "<answer>Paris</answer>",
        ])

        def capture_and_respond(messages, temperature, max_new_tokens):
            injected_messages.extend(messages)
            return next(turn_responses)

        with patch.object(model, "_call_model", side_effect=capture_and_respond), \
             patch("src.models.deepeyes_v2._execute_code", return_value="hello"):
            model._run_chain(TINY_IMAGE, QUERY, temperature=0.0, max_new_tokens=512)

        tool_result_msgs = [
            m for m in injected_messages
            if isinstance(m.get("content"), str) and "Code execution result:" in m["content"]
        ]
        assert len(tool_result_msgs) >= 1, "No 'Code execution result:' message found"
        content = tool_result_msgs[0]["content"]
        # Exact paper format: triple-quote delimiters on their own lines.
        assert "stdout:" in content
        assert "stderr:" in content
        assert "'''" in content

    def test_run_chain_user_prompt_contains_answer_tag_instruction(self):
        """The user message must include the <answer> tag instruction from USER_PROMPT_TEMPLATE."""
        from unittest.mock import patch

        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = self._make_model()
        captured_messages: list = []

        def capture(messages, temperature, max_new_tokens):
            captured_messages.extend(messages)
            return "<answer>X</answer>"

        with patch.object(model, "_call_model", side_effect=capture):
            model._run_chain(TINY_IMAGE, QUERY, temperature=0.0, max_new_tokens=512)

        # Find the first user message (the one with the query).
        user_msgs = [m for m in captured_messages if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        first_user_content = user_msgs[0]["content"]
        # Content is a list (image + text) for the first turn.
        text_parts = [
            part["text"] for part in first_user_content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        assert any("<answer>" in t for t in text_parts), (
            "USER_PROMPT_TEMPLATE must inject '<answer>' instruction into user message"
        )

    # ---- termination condition ---------------------------------------------

    def test_run_chain_terminates_on_answer_tag(self):
        from unittest.mock import patch

        import numpy as np

        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = self._make_model()
        with patch.object(model, "_call_model", return_value="<answer>Madrid</answer>"):
            chain = model._run_chain(TINY_IMAGE, QUERY, temperature=0.0, max_new_tokens=512)
        assert chain["answer"] == "Madrid"
        assert len(chain["cot_steps"]) >= 1

    def test_run_chain_answer_is_empty_string_when_no_tag_after_max_turns(self):
        from unittest.mock import patch

        from src.models.deepeyes_v2 import DeepEyesV2Model, _execute_code

        model = self._make_model()
        model.max_turns = 2
        with patch.object(model, "_call_model", return_value="<code>print(1)</code>"), \
             patch("src.models.deepeyes_v2._execute_code", return_value="1"):
            chain = model._run_chain(TINY_IMAGE, QUERY, temperature=0.0, max_new_tokens=512)
        assert chain["answer"] == ""
        assert len(chain["cot_steps"]) == 2

    # ---- lazy loading guard ------------------------------------------------

    def test_generate_calls_load_before_chain(self):
        from src.models.deepeyes_v2 import DeepEyesV2Model

        model = DeepEyesV2Model.__new__(DeepEyesV2Model)
        model.model_id = "honglyhly/DeepEyesV2_7B_1031"
        model.max_turns = 10
        model._model = None
        model._processor = None
        stub = {"bbox_raw": None, "coords": [], "answer": "X", "cot_steps": [], "tool_results": []}
        with patch.object(model, "_load") as mock_load, \
             patch.object(model, "_run_chain", return_value=stub):
            model.generate(TINY_IMAGE, QUERY, n=1)
        mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# TestGRITModel
# ---------------------------------------------------------------------------


class TestGRITModel:
    """Tests for GRITModel — all model weights are mocked."""

    @staticmethod
    def _make_model() -> "GRITModel":  # type: ignore[name-defined]
        from src.models.grit import GRITModel

        model = GRITModel.__new__(GRITModel)
        model.model_id = "yfan1997/GRIT-20-Qwen2.5-VL-3B"
        model.load_in_8bit = False
        model._model = MagicMock()
        model._processor = MagicMock()
        model._processor.apply_chat_template.return_value = "fake prompt"
        model._processor.return_value = MagicMock(
            **{"__getitem__.return_value": MagicMock(),
               "to.return_value": {"input_ids": MagicMock(shape=(1, 10))}}
        )
        model._processor.batch_decode.return_value = [
            "<think>some reasoning</think><rethink>rethink</rethink><answer>Paris</answer>"
        ]
        model._model.generate.return_value = MagicMock()
        model._model.device = "cpu"
        return model

    # ---- answer parser ------------------------------------------------------

    def test_parse_grit_answer_extracts_text_between_tags(self) -> None:
        from src.models.grit import _parse_grit_answer

        assert _parse_grit_answer("<answer>London</answer>") == "London"

    def test_parse_grit_answer_returns_none_when_no_tag(self) -> None:
        from src.models.grit import _parse_grit_answer

        assert _parse_grit_answer("The answer is Paris.") is None

    def test_parse_grit_answer_handles_missing_closing_tag(self) -> None:
        from src.models.grit import _parse_grit_answer

        assert _parse_grit_answer("<answer>blue") == "blue"

    def test_parse_grit_answer_strips_whitespace(self) -> None:
        from src.models.grit import _parse_grit_answer

        assert _parse_grit_answer("<answer>  Tokyo  </answer>") == "Tokyo"

    def test_parse_grit_answer_handles_full_grit_output(self) -> None:
        from src.models.grit import _parse_grit_answer

        text = (
            "<think>Look at the image carefully.</think>"
            '{"bbox_2d": [10, 20, 100, 200]}'
            "<rethink>It is blue.</rethink>"
            "<answer>blue</answer>"
        )
        assert _parse_grit_answer(text) == "blue"

    # ---- generate contract --------------------------------------------------

    def test_generate_returns_list_of_n_dicts(self) -> None:
        from src.models.grit import GRITModel

        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "Rome",
                "cot_steps": [], "tool_results": []}
        with patch.object(model, "_answer", return_value=stub):
            chains = model.generate(TINY_IMAGE, QUERY, n=3)
        assert len(chains) == 3

    def test_generate_chain_has_required_keys(self) -> None:
        from src.models.grit import GRITModel

        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "Berlin",
                "cot_steps": ["think"], "tool_results": []}
        with patch.object(model, "_answer", return_value=stub):
            chain = model.generate(TINY_IMAGE, QUERY, n=1)[0]
        for key in ("bbox_raw", "coords", "answer", "cot_steps", "tool_results"):
            assert key in chain

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_generate_calls_answer_exactly_n_times(self, n: int) -> None:
        from src.models.grit import GRITModel

        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "X",
                "cot_steps": [], "tool_results": []}
        with patch.object(model, "_answer", return_value=stub) as mock_ans:
            model.generate(TINY_IMAGE, QUERY, n=n)
        assert mock_ans.call_count == n

    def test_generate_calls_load_before_answer(self) -> None:
        from src.models.grit import GRITModel

        model = GRITModel.__new__(GRITModel)
        model.model_id = "yfan1997/GRIT-20-Qwen2.5-VL-3B"
        model.load_in_8bit = False
        model._model = None
        model._processor = None
        stub = {"bbox_raw": None, "coords": [], "answer": "X",
                "cot_steps": [], "tool_results": []}
        with patch.object(model, "_load") as mock_load, \
             patch.object(model, "_answer", return_value=stub):
            model.generate(TINY_IMAGE, QUERY, n=1)
        mock_load.assert_called_once()

    def test_generate_cot_steps_contains_think_content(self) -> None:
        from src.models.grit import GRITModel

        model = self._make_model()
        stub = {"bbox_raw": None, "coords": [], "answer": "blue",
                "cot_steps": ["Look at the image carefully."], "tool_results": []}
        with patch.object(model, "_answer", return_value=stub):
            chain = model.generate(TINY_IMAGE, QUERY, n=1)[0]
        assert len(chain["cot_steps"]) >= 1
