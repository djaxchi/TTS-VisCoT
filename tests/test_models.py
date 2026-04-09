"""Tests for DeepEyesV2Model — pure unit tests, no GPU or model downloads."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models.deepeyes_v2 import (
    DeepEyesV2Model,
    _execute_code,
    _extract_code_block,
    _extract_tool_call,
    _fix_python_indentation,
    _parse_answer,
    RETURN_SEARCH_PROMPT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 64, 32))


# ---------------------------------------------------------------------------
# _parse_answer
# ---------------------------------------------------------------------------


def test_parse_answer_returns_stripped_content():
    assert _parse_answer("<answer>  parrot  </answer>") == "parrot"


def test_parse_answer_multiline_content():
    assert _parse_answer("<answer>\nfoo\nbar\n</answer>") == "foo\nbar"


def test_parse_answer_returns_none_when_missing():
    assert _parse_answer("I think it is a parrot.") is None


def test_parse_answer_returns_first_tag_only():
    assert _parse_answer("<answer>first</answer> <answer>second</answer>") == "first"


# ---------------------------------------------------------------------------
# _extract_code_block
# ---------------------------------------------------------------------------


def test_extract_code_block_backtick_fence_inside_code_tag():
    text = "<code>\n```python\nprint(1+1)\n```\n</code>"
    assert _extract_code_block(text) == "print(1+1)"


def test_extract_code_block_triple_quote_inside_code_tag():
    text = "<code>\n'''python\nprint('hi')\n'''\n</code>"
    assert _extract_code_block(text) == "print('hi')"


def test_extract_code_block_bare_code_tag():
    text = "<code>x = 1\nprint(x)</code>"
    assert _extract_code_block(text) == "x = 1\nprint(x)"


def test_extract_code_block_bare_backtick_fence():
    text = "Some text\n```python\nresult = 42\n```\nmore text"
    assert _extract_code_block(text) == "result = 42"


def test_extract_code_block_returns_none_when_missing():
    assert _extract_code_block("<answer>yes</answer>") is None


def test_extract_code_block_returns_last_when_multiple():
    text = (
        "<code>\n```python\nprint('first')\n```\n</code>\n"
        "<code>\n```python\nprint('last')\n```\n</code>"
    )
    assert _extract_code_block(text) == "print('last')"


# ---------------------------------------------------------------------------
# _extract_tool_call
# ---------------------------------------------------------------------------


def test_extract_tool_call_search_with_query():
    text = '<tool_call>{"name": "search", "arguments": {"query": "Eiffel Tower"}}</tool_call>'
    result = _extract_tool_call(text)
    assert result == {"name": "search", "arguments": {"query": "Eiffel Tower"}}


def test_extract_tool_call_image_search_no_args():
    text = '<tool_call>{"name": "image_search"}</tool_call>'
    result = _extract_tool_call(text)
    assert result == {"name": "image_search"}


def test_extract_tool_call_returns_none_when_missing():
    assert _extract_tool_call("<code>print(1)</code>") is None


def test_extract_tool_call_returns_last_when_multiple():
    text = (
        '<tool_call>{"name": "search", "arguments": {"query": "first"}}</tool_call>\n'
        '<tool_call>{"name": "search", "arguments": {"query": "last"}}</tool_call>'
    )
    result = _extract_tool_call(text)
    assert result["arguments"]["query"] == "last"


def test_extract_tool_call_returns_none_on_invalid_json():
    assert _extract_tool_call("<tool_call>not json</tool_call>") is None


# ---------------------------------------------------------------------------
# _fix_python_indentation
# ---------------------------------------------------------------------------


def test_fix_python_indentation_removes_uniform_leading_indent():
    code = "    x = 1\n    print(x)"
    assert _fix_python_indentation(code) == "x = 1\nprint(x)"


def test_fix_python_indentation_strips_outer_whitespace():
    assert _fix_python_indentation("\n  x = 1\n") == "x = 1"


def test_fix_python_indentation_preserves_relative_indent():
    code = "    if True:\n        x = 1"
    result = _fix_python_indentation(code)
    assert result == "if True:\n    x = 1"


# ---------------------------------------------------------------------------
# _execute_code
# ---------------------------------------------------------------------------


def test_execute_code_captures_stdout():
    stdout, stderr, figs = _execute_code("print('hello')", {})
    assert stdout == "hello"
    assert stderr == ""
    assert figs == []


def test_execute_code_returns_error_on_exception():
    stdout, stderr, figs = _execute_code("raise ValueError('oops')", {})
    assert "ValueError" in stderr
    assert "oops" in stderr


def test_execute_code_namespace_persists():
    ns: Dict[str, Any] = {}
    _execute_code("x = 42", ns)
    stdout, stderr, _ = _execute_code("print(x)", ns)
    assert stdout == "42"
    assert stderr == ""


def test_execute_code_image_available_in_namespace():
    img = _tiny_image()
    ns: Dict[str, Any] = {"image_1": img}
    stdout, stderr, _ = _execute_code("print(image_1.size)", ns)
    assert stdout == "(64, 64)"
    assert stderr == ""


def test_execute_code_returns_empty_figures_without_matplotlib_show():
    stdout, stderr, figs = _execute_code("x = 1", {})
    assert figs == []


# ---------------------------------------------------------------------------
# DeepEyesV2Model.generate — mocked _call_model
# ---------------------------------------------------------------------------


def _make_model() -> DeepEyesV2Model:
    model = DeepEyesV2Model.__new__(DeepEyesV2Model)
    model.model_id = "test"
    model.max_turns = 10
    model.load_in_8bit = False
    model._model = MagicMock()
    model._processor = MagicMock()
    return model


def test_generate_returns_n_chain_dicts():
    model = _make_model()
    model._call_model = MagicMock(return_value="<answer>parrot</answer>")
    results = model.generate(_tiny_image(), "What bird is this?", n=3)
    assert len(results) == 3


def test_generate_chain_dict_has_required_keys():
    model = _make_model()
    model._call_model = MagicMock(return_value="<answer>yes</answer>")
    result = model.generate(_tiny_image(), "Is this indoors?", n=1)[0]
    assert set(result.keys()) >= {"bbox_raw", "coords", "answer", "cot_steps", "tool_results"}


def test_generate_answer_extracted_from_tag():
    model = _make_model()
    model._call_model = MagicMock(return_value="<answer>parrot</answer>")
    result = model.generate(_tiny_image(), "What bird?", n=1)[0]
    assert result["answer"] == "parrot"


def test_generate_bbox_raw_is_none_and_coords_empty():
    model = _make_model()
    model._call_model = MagicMock(return_value="<answer>blue</answer>")
    result = model.generate(_tiny_image(), "What color?", n=1)[0]
    assert result["bbox_raw"] is None
    assert result["coords"] == []


def test_generate_cot_steps_records_all_turns():
    model = _make_model()
    responses = [
        "<code>\n```python\nprint(1)\n```\n</code>",
        "<answer>done</answer>",
    ]
    model._call_model = MagicMock(side_effect=responses)
    result = model.generate(_tiny_image(), "Count?", n=1)[0]
    assert len(result["cot_steps"]) == 2
    assert result["answer"] == "done"


def test_generate_tool_results_recorded_after_code_execution():
    model = _make_model()
    responses = [
        "<code>\n```python\nprint('output')\n```\n</code>",
        "<answer>42</answer>",
    ]
    model._call_model = MagicMock(side_effect=responses)
    result = model.generate(_tiny_image(), "What?", n=1)[0]
    assert len(result["tool_results"]) == 1
    assert "output" in result["tool_results"][0]


def test_generate_max_turns_exhausted_returns_empty_answer():
    model = _make_model()
    model.max_turns = 3
    # Always return a code block — never terminates with <answer>
    model._call_model = MagicMock(
        return_value="<code>\n```python\nprint(1)\n```\n</code>"
    )
    result = model.generate(_tiny_image(), "Loop?", n=1)[0]
    assert result["answer"] == ""


def test_generate_tool_call_search_returns_stub_and_continues():
    """Model emits <tool_call> search — pipeline stubs result and continues to <answer>."""
    model = _make_model()
    responses = [
        '<tool_call>{"name": "search", "arguments": {"query": "what is a parrot"}}</tool_call>',
        "<answer>parrot</answer>",
    ]
    model._call_model = MagicMock(side_effect=responses)
    result = model.generate(_tiny_image(), "What bird?", n=1)[0]
    assert result["answer"] == "parrot"
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0] == RETURN_SEARCH_PROMPT


def test_generate_tool_call_image_search_returns_stub_and_continues():
    """Model emits <tool_call> image_search — same stub, chain continues."""
    model = _make_model()
    responses = [
        '<tool_call>{"name": "image_search"}</tool_call>',
        "<answer>cat</answer>",
    ]
    model._call_model = MagicMock(side_effect=responses)
    result = model.generate(_tiny_image(), "What animal?", n=1)[0]
    assert result["answer"] == "cat"
    assert result["tool_results"][0] == RETURN_SEARCH_PROMPT


def test_generate_independent_namespaces_across_chains():
    """Each chain gets its own exec namespace — variables don't leak between chains."""
    model = _make_model()
    call_count = 0

    def side_effect(messages, temp, tokens):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return "<code>\n```python\nx = 99\n```\n</code>"
        return "<answer>done</answer>"

    model._call_model = MagicMock(side_effect=side_effect)
    results = model.generate(_tiny_image(), "Test?", n=2)
    # Both chains should complete without cross-namespace bleed
    assert all(r["answer"] == "done" for r in results)
