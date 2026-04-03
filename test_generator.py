"""
Unit tests for the TraceAudit generator module.
Run with: pytest tests/ -v
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import MutationRecord, BUG_LABELS, ALL_BUG_TYPES


class TestMutationRecord:
    """Validate Pydantic schema enforcement."""

    def _base_record(self) -> dict:
        return {
            "original_function_name": "test_fn",
            "original_code": "def test_fn():\n    return 42\n",
            "mutated_code": "def test_fn():\n    return 43\n",
            "bug_type": "BOUNDARY_VIOLATION",
            "bug_line_number": 2,
            "bug_description": "Changed return value from 42 to 43.",
            "is_detectable_by_linter": False,
            "difficulty": "easy",
        }

    def test_valid_record(self):
        r = MutationRecord(**self._base_record())
        assert r.bug_type == "BOUNDARY_VIOLATION"
        assert r.bug_line_number == 2

    def test_invalid_bug_type(self):
        data = self._base_record()
        data["bug_type"] = "FAKE_BUG"
        with pytest.raises(Exception):
            MutationRecord(**data)

    def test_syntax_error_rejected(self):
        data = self._base_record()
        data["mutated_code"] = "def broken(:\n    pass\n"
        with pytest.raises(Exception):
            MutationRecord(**data)

    def test_line_number_must_be_positive(self):
        data = self._base_record()
        data["bug_line_number"] = 0
        with pytest.raises(Exception):
            MutationRecord(**data)

    def test_all_bug_types_are_valid(self):
        data = self._base_record()
        for bug_type in ALL_BUG_TYPES:
            data["bug_type"] = bug_type
            r = MutationRecord(**data)
            assert r.bug_type == bug_type


class TestSeedFunctions:
    """Verify the seed vault loads correctly."""

    def test_seeds_load(self):
        from data.seeds import SEED_FUNCTIONS
        assert len(SEED_FUNCTIONS) >= 15, "Need at least 15 seed functions"

    def test_seeds_have_required_fields(self):
        from data.seeds import SEED_FUNCTIONS
        for seed in SEED_FUNCTIONS:
            assert "name" in seed
            assert "source" in seed
            assert "docstring" in seed
            assert len(seed["source"]) > 0

    def test_seed_sources_are_valid_python(self):
        from data.seeds import SEED_FUNCTIONS
        for seed in SEED_FUNCTIONS:
            try:
                compile(seed["source"], seed["name"], "exec")
            except SyntaxError as e:
                pytest.fail(f"Seed '{seed['name']}' has SyntaxError: {e}")


class TestStyleGuide:
    """Verify style_guide.json is well-formed."""

    def test_style_guide_loads(self):
        guide_path = PROJECT_ROOT / "data" / "style_guide.json"
        with guide_path.open() as f:
            guide = json.load(f)
        assert "bug_categories" in guide

    def test_all_bug_types_in_guide(self):
        guide_path = PROJECT_ROOT / "data" / "style_guide.json"
        with guide_path.open() as f:
            guide = json.load(f)
        for bug_type in ALL_BUG_TYPES:
            assert bug_type in guide["bug_categories"], f"{bug_type} missing from style_guide.json"

    def test_guide_has_required_fields(self):
        guide_path = PROJECT_ROOT / "data" / "style_guide.json"
        with guide_path.open() as f:
            guide = json.load(f)
        required = ["detection_signals", "trace_checklist", "description"]
        for bug_type, cat in guide["bug_categories"].items():
            for field in required:
                assert field in cat, f"{bug_type} missing '{field}' in style_guide.json"
            assert len(cat["detection_signals"]) >= 3
            assert len(cat["trace_checklist"]) >= 3
