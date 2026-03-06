"""
Tests for RAG-derived baker's percentage extraction.

Covers:
  - _parse_grams()                          — gram amount string parsing
  - _classify_ingredient()                  — ingredient name classification
  - _parse_baker_pcts_from_ingredients()    — full baker's % computation from ingredient list
  - _extract_pcts_from_docs()               — LLM-based extraction from retrieved docs (mocked)
  - compute_baking_math()                   — integration: RAG pcts flow through to math results
  - build_timeline() RAG back-fill          — integration: extracted ingredients update starter/salt pcts
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ── helpers from bake_plan ────────────────────────────────────────────────────
from app.graph.nodes.bake_plan import (
    _classify_ingredient,
    _parse_grams,
    _parse_baker_pcts_from_ingredients,
)

# ── helpers from recipe ───────────────────────────────────────────────────────
from app.graph.nodes.recipe import (
    _extract_pcts_from_docs,
    compute_baking_math,
)


# =============================================================================
# _parse_grams
# =============================================================================


class TestParseGrams:
    def test_plain_grams(self):
        assert _parse_grams("300g") == 300.0

    def test_grams_with_space(self):
        assert _parse_grams("300 g") == 300.0

    def test_no_unit(self):
        # bare number treated as grams
        assert _parse_grams("500") == 500.0

    def test_kilograms(self):
        assert _parse_grams("0.5kg") == 500.0

    def test_kilograms_with_space(self):
        assert _parse_grams("1 kg") == 1000.0

    def test_tilde_prefix(self):
        assert _parse_grams("~100g") == 100.0

    def test_decimal(self):
        assert _parse_grams("12.5g") == 12.5

    def test_volume_tsp_returns_none(self):
        assert _parse_grams("2 tsp") is None

    def test_volume_cup_returns_none(self):
        assert _parse_grams("1 cup") is None

    def test_volume_ml_returns_none(self):
        assert _parse_grams("250ml") is None

    def test_volume_oz_returns_none(self):
        assert _parse_grams("4 oz") is None

    def test_empty_string_returns_none(self):
        assert _parse_grams("") is None

    def test_none_returns_none(self):
        assert _parse_grams(None) is None  # type: ignore[arg-type]

    def test_non_numeric_string_returns_none(self):
        assert _parse_grams("a pinch") is None


# =============================================================================
# _classify_ingredient
# =============================================================================


class TestClassifyIngredient:
    # Flour variants
    def test_flour(self):
        assert _classify_ingredient("bread flour") == "flour"

    def test_rye_flour(self):
        assert _classify_ingredient("dark rye flour") == "flour"

    def test_whole_wheat(self):
        assert _classify_ingredient("whole wheat flour") == "flour"

    def test_spelt(self):
        assert _classify_ingredient("spelt flour") == "flour"

    # Starter variants
    def test_starter(self):
        assert _classify_ingredient("sourdough starter") == "starter"

    def test_levain(self):
        assert _classify_ingredient("levain") == "starter"

    def test_preferment(self):
        assert _classify_ingredient("preferment") == "starter"

    def test_leaven(self):
        assert _classify_ingredient("leaven") == "starter"

    # Salt
    def test_salt(self):
        assert _classify_ingredient("salt") == "salt"

    def test_sea_salt(self):
        assert _classify_ingredient("sea salt") == "salt"

    # Water
    def test_water(self):
        assert _classify_ingredient("water") == "water"

    def test_milk(self):
        assert _classify_ingredient("whole milk") == "water"

    # Other
    def test_seeds(self):
        assert _classify_ingredient("sesame seeds") == "other"

    def test_butter(self):
        assert _classify_ingredient("butter") == "other"

    # Salt wins over flour (edge case: "salted flour" would be a weird name, but test priority)
    def test_salt_keyword_priority(self):
        # "salt" is checked before "flour", so a hypothetical "salt flour" → salt
        assert _classify_ingredient("salt") == "salt"


# =============================================================================
# _parse_baker_pcts_from_ingredients
# =============================================================================


class TestParseBakerPctsFromIngredients:
    def _make(self, **kwargs):
        """Build ingredient list from name=amount kwargs."""
        return [{"name": k, "amount": v} for k, v in kwargs.items()]

    def test_typical_recipe(self):
        """500g flour, 100g starter (20%), 10g salt (2%)."""
        ings = self._make(
            **{
                "bread flour": "500g",
                "sourdough starter": "100g",
                "salt": "10g",
                "water": "375g",
            }
        )
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] == pytest.approx(20.0, abs=0.1)
        assert result["salt_pct"] == pytest.approx(2.0, abs=0.1)

    def test_rye_recipe(self):
        """300g rye + 200g bread flour = 500g total; 75g levain (15%), 10g salt (2%)."""
        ings = [
            {"name": "dark rye flour", "amount": "300g"},
            {"name": "bread flour", "amount": "200g"},
            {"name": "levain", "amount": "75g"},
            {"name": "salt", "amount": "10g"},
            {"name": "water", "amount": "400g"},
        ]
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] == pytest.approx(15.0, abs=0.1)
        assert result["salt_pct"] == pytest.approx(2.0, abs=0.1)

    def test_no_flour_returns_none(self):
        """If flour weight cannot be parsed, both pcts must be None."""
        ings = self._make(
            **{"sourdough starter": "100g", "salt": "10g", "water": "375g"}
        )
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None

    def test_no_starter_returns_none_for_starter(self):
        ings = self._make(**{"bread flour": "500g", "salt": "10g", "water": "375g"})
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] is None
        assert result["salt_pct"] == pytest.approx(2.0, abs=0.1)

    def test_no_salt_returns_none_for_salt(self):
        ings = self._make(**{"bread flour": "500g", "levain": "100g", "water": "375g"})
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] == pytest.approx(20.0, abs=0.1)
        assert result["salt_pct"] is None

    def test_volume_amounts_ignored(self):
        """Ingredients with unparseable volume amounts don't crash and don't contribute."""
        ings = [
            {"name": "bread flour", "amount": "500g"},
            {"name": "sourdough starter", "amount": "2 cups"},  # volume → ignored
            {"name": "salt", "amount": "10g"},
        ]
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] is None  # starter amount ignored
        assert result["salt_pct"] == pytest.approx(2.0, abs=0.1)

    def test_empty_list_returns_nones(self):
        result = _parse_baker_pcts_from_ingredients([])
        assert result == {"starter_pct": None, "salt_pct": None}

    def test_kg_amounts(self):
        """1kg flour, 200g starter = 20%, 20g salt = 2%."""
        ings = self._make(
            **{"flour": "1kg", "sourdough starter": "200g", "salt": "20g"}
        )
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] == pytest.approx(20.0, abs=0.1)
        assert result["salt_pct"] == pytest.approx(2.0, abs=0.1)

    def test_multiple_flour_types_summed(self):
        """Two flour types should be summed to get total flour weight."""
        ings = [
            {"name": "whole wheat flour", "amount": "300g"},
            {"name": "rye flour", "amount": "200g"},
            {"name": "levain", "amount": "100g"},  # 100/500 = 20%
            {"name": "salt", "amount": "9g"},  # 9/500  = 1.8%
        ]
        result = _parse_baker_pcts_from_ingredients(ings)
        assert result["starter_pct"] == pytest.approx(20.0, abs=0.1)
        assert result["salt_pct"] == pytest.approx(1.8, abs=0.1)

    def test_rounding(self):
        """Results should be rounded to 1 decimal place."""
        ings = self._make(**{"flour": "300g", "levain": "55g", "salt": "7g"})
        result = _parse_baker_pcts_from_ingredients(ings)
        # 55/300*100 = 18.333... → 18.3
        assert result["starter_pct"] == pytest.approx(18.3, abs=0.05)
        # 7/300*100  = 2.333...  → 2.3
        assert result["salt_pct"] == pytest.approx(2.3, abs=0.05)


# =============================================================================
# _extract_pcts_from_docs  (recipe path — LLM mocked)
# =============================================================================


class TestExtractPctsFromDocs:
    DOCS = [
        {"source": "Book.pdf", "text": "Use 20% starter and 2% salt.", "score": 0.9}
    ]

    def _mock_llm_response(self, content: str):
        """Return a mock LLM that produces the given content string."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = content
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("app.graph.nodes.recipe.get_llm")
    def test_extracts_both_pcts(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response(
            '{"starter_pct": 20.0, "salt_pct": 2.0}'
        )
        result = _extract_pcts_from_docs(self.DOCS, "country loaf")
        assert result["starter_pct"] == pytest.approx(20.0)
        assert result["salt_pct"] == pytest.approx(2.0)

    @patch("app.graph.nodes.recipe.get_llm")
    def test_null_values_returned_as_none(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response(
            '{"starter_pct": null, "salt_pct": null}'
        )
        result = _extract_pcts_from_docs(self.DOCS, "focaccia")
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None

    @patch("app.graph.nodes.recipe.get_llm")
    def test_strips_markdown_fences(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response(
            '```json\n{"starter_pct": 15.0, "salt_pct": 1.8}\n```'
        )
        result = _extract_pcts_from_docs(self.DOCS, "rye bread")
        assert result["starter_pct"] == pytest.approx(15.0)
        assert result["salt_pct"] == pytest.approx(1.8)

    @patch("app.graph.nodes.recipe.get_llm")
    def test_invalid_json_returns_nones(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response("not json at all")
        result = _extract_pcts_from_docs(self.DOCS, "ciabatta")
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None

    @patch("app.graph.nodes.recipe.get_llm")
    def test_llm_exception_returns_nones(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_get_llm.return_value = mock_llm
        result = _extract_pcts_from_docs(self.DOCS, "ciabatta")
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None

    def test_empty_docs_returns_nones_without_llm_call(self):
        """No LLM call should be made when docs list is empty."""
        result = _extract_pcts_from_docs([], "focaccia")
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None

    @patch("app.graph.nodes.recipe.get_llm")
    def test_zero_value_treated_as_none(self, mock_get_llm):
        """A returned value of 0 is not a valid percentage — treat as None."""
        mock_get_llm.return_value = self._mock_llm_response(
            '{"starter_pct": 0, "salt_pct": 0}'
        )
        result = _extract_pcts_from_docs(self.DOCS, "bread")
        assert result["starter_pct"] is None
        assert result["salt_pct"] is None


# =============================================================================
# compute_baking_math — integration: RAG pcts flow through
# =============================================================================


class TestComputeBakingMathRagIntegration:
    """Test that compute_baking_math picks up RAG-derived pcts when user hasn't provided them."""

    DOCS = [{"source": "Book.pdf", "text": "15% levain, 1.8% salt.", "score": 0.9}]

    def _base_state(self, params: dict, docs: list = None) -> dict:
        return {
            "intent_params": params,
            "retrieved_docs": docs if docs is not None else self.DOCS,
            "messages": [],
            "user_query": "give me a recipe",
        }

    @patch("app.graph.nodes.recipe.get_llm")
    def test_rag_pcts_used_when_user_omits_them(self, mock_get_llm):
        """When user provides no starter_pct/salt_pct, RAG values should be used."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"starter_pct": 15.0, "salt_pct": 1.8}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._base_state(
            params={"target_product": "rye bread", "hydration": 80.0, "flour_g": 500.0}
        )
        result = compute_baking_math(state)
        math = result["math_results"]

        expected_starter_g = 500.0 * 15.0 / 100  # 75g
        expected_salt_g = 500.0 * 1.8 / 100  # 9g
        assert math["ingredients"]["starter"] == pytest.approx(
            expected_starter_g, abs=0.5
        )
        assert math["ingredients"]["salt"] == pytest.approx(expected_salt_g, abs=0.5)
        assert math["bakers_percentages"]["starter"] == pytest.approx(15.0, abs=0.5)
        assert math["bakers_percentages"]["salt"] == pytest.approx(1.8, abs=0.5)

    @patch("app.graph.nodes.recipe.get_llm")
    def test_user_value_beats_rag(self, mock_get_llm):
        """User-supplied starter_pct/salt_pct must override RAG-derived values."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # RAG says 15% / 1.8%, but user said 25% / 3%
        mock_response.content = '{"starter_pct": 15.0, "salt_pct": 1.8}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._base_state(
            params={
                "target_product": "rye bread",
                "hydration": 80.0,
                "flour_g": 500.0,
                "starter_pct": 25.0,
                "salt_pct": 3.0,
            }
        )
        result = compute_baking_math(state)
        math = result["math_results"]

        expected_starter_g = 500.0 * 25.0 / 100  # 125g
        expected_salt_g = 500.0 * 3.0 / 100  # 15g
        assert math["ingredients"]["starter"] == pytest.approx(
            expected_starter_g, abs=0.5
        )
        assert math["ingredients"]["salt"] == pytest.approx(expected_salt_g, abs=0.5)

    @patch("app.graph.nodes.recipe.get_llm")
    def test_rag_null_falls_back_to_config_default(self, mock_get_llm):
        """If RAG returns null for a pct, the bread config default must be used."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"starter_pct": null, "salt_pct": null}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self._base_state(
            params={
                "target_product": "country loaf",
                "hydration": 75.0,
                "flour_g": 1000.0,
            }
        )
        result = compute_baking_math(state)
        math = result["math_results"]

        # country_loaf defaults: starter=20%, salt=2%
        assert math["ingredients"]["starter"] == pytest.approx(200.0, abs=1.0)
        assert math["ingredients"]["salt"] == pytest.approx(20.0, abs=1.0)

    def test_no_docs_falls_back_to_config_default(self):
        """With no retrieved docs, LLM is never called; config defaults are used."""
        state = self._base_state(
            params={
                "target_product": "country loaf",
                "hydration": 75.0,
                "flour_g": 1000.0,
            },
            docs=[],
        )
        result = compute_baking_math(state)
        math = result["math_results"]

        assert math["ingredients"]["starter"] == pytest.approx(200.0, abs=1.0)
        assert math["ingredients"]["salt"] == pytest.approx(20.0, abs=1.0)

    @patch("app.graph.nodes.recipe.get_llm")
    def test_no_rag_call_when_both_params_present(self, mock_get_llm):
        """LLM should not be called when both starter_pct and salt_pct are already in params."""
        state = self._base_state(
            params={
                "target_product": "focaccia",
                "hydration": 75.0,
                "flour_g": 500.0,
                "starter_pct": 20.0,
                "salt_pct": 2.5,
            }
        )
        compute_baking_math(state)
        # LLM must not have been called
        mock_get_llm.assert_not_called()


# =============================================================================
# build_timeline — RAG pct back-fill integration
# =============================================================================


class TestBuildTimelineRagBackfill:
    """Test that build_timeline back-fills starter_pct/salt_pct from extracted ingredients."""

    def _make_state(
        self, params: dict, extracted_ingredients: list | None = None
    ) -> dict:
        """
        Build a minimal state for build_timeline.
        We mock _extract_recipe_from_docs so no LLM call is needed.
        """
        return {
            "intent_params": params,
            "retrieved_docs": [
                {"source": "Book.pdf", "text": "rye bread recipe", "score": 0.9}
            ],
            "messages": [],
            "user_query": "plan my bake",
            "_mock_ingredients": extracted_ingredients,
        }

    @patch("app.graph.nodes.bake_plan._extract_recipe_from_docs")
    def test_rag_starter_and_salt_back_filled(self, mock_extract):
        """When user doesn't set starter/salt, extracted ingredients should drive the pcts."""
        from app.graph.nodes.bake_plan import build_timeline

        mock_extract.return_value = {
            "recipe_found": True,
            "ingredients": [
                {"name": "dark rye flour", "amount": "500g"},
                {"name": "levain", "amount": "75g"},  # 15%
                {"name": "salt", "amount": "9g"},  # 1.8%
                {"name": "water", "amount": "425g"},
            ],
            "steps": [
                {"name": "Mix", "duration_minutes": 15, "description": "Mix dough"},
                {
                    "name": "Bulk ferment",
                    "duration_minutes": 240,
                    "description": "Ferment",
                },
                {"name": "Bake", "duration_minutes": 50, "description": "Bake"},
            ],
        }

        state = {
            "intent_params": {
                "target_product": "rye bread",
                "num_loaves": 1,
                "temperature_c": 22.0,
                "start_time": "2026-01-01T09:00:00",
            },
            "retrieved_docs": [{"source": "Book.pdf", "text": "rye", "score": 0.9}],
            "messages": [],
            "user_query": "plan rye bake",
        }

        result = build_timeline(state)
        plan = result["bake_plan_data"]

        # compute_recipe uses the back-filled starter_pct (15%) not the default (20%)
        recipe = plan["recipe"]
        flour_g = recipe["flour_g"]
        assert recipe["starter_g"] == pytest.approx(flour_g * 0.15, abs=2.0)
        assert recipe["salt_g"] == pytest.approx(flour_g * 0.018, abs=1.0)

    @patch("app.graph.nodes.bake_plan._extract_recipe_from_docs")
    def test_user_value_beats_rag_in_build_timeline(self, mock_extract):
        """Explicitly provided starter_pct/salt_pct must not be overridden by RAG."""
        from app.graph.nodes.bake_plan import build_timeline

        mock_extract.return_value = {
            "recipe_found": True,
            "ingredients": [
                {"name": "dark rye flour", "amount": "500g"},
                {"name": "levain", "amount": "75g"},  # RAG says 15%
                {"name": "salt", "amount": "9g"},  # RAG says 1.8%
            ],
            "steps": [
                {"name": "Mix", "duration_minutes": 15, "description": "Mix"},
                {"name": "Bulk", "duration_minutes": 240, "description": "Ferment"},
                {"name": "Bake", "duration_minutes": 50, "description": "Bake"},
            ],
        }

        state = {
            "intent_params": {
                "target_product": "rye bread",
                "num_loaves": 1,
                "temperature_c": 22.0,
                "start_time": "2026-01-01T09:00:00",
                "starter_pct": 25.0,  # user explicitly set
                "salt_pct": 3.0,  # user explicitly set
            },
            "retrieved_docs": [{"source": "Book.pdf", "text": "rye", "score": 0.9}],
            "messages": [],
            "user_query": "plan rye bake",
        }

        result = build_timeline(state)
        recipe = result["bake_plan_data"]["recipe"]
        flour_g = recipe["flour_g"]

        # Should use 25% / 3%, not 15% / 1.8%
        assert recipe["starter_g"] == pytest.approx(flour_g * 0.25, abs=2.0)
        assert recipe["salt_g"] == pytest.approx(flour_g * 0.03, abs=1.0)

    @patch("app.graph.nodes.bake_plan._extract_recipe_from_docs")
    def test_no_flour_in_ingredients_falls_back_to_default(self, mock_extract):
        """If extracted ingredients have no flour, _parse_baker_pcts returns None → use defaults."""
        from app.graph.nodes.bake_plan import build_timeline

        mock_extract.return_value = {
            "recipe_found": True,
            "ingredients": [
                # No flour entry → can't compute percentages
                {"name": "levain", "amount": "100g"},
                {"name": "salt", "amount": "10g"},
            ],
            "steps": [
                {"name": "Mix", "duration_minutes": 15, "description": "Mix"},
                {"name": "Bulk", "duration_minutes": 240, "description": "Ferment"},
                {"name": "Bake", "duration_minutes": 50, "description": "Bake"},
            ],
        }

        state = {
            "intent_params": {
                "target_product": "rye bread",
                "num_loaves": 1,
                "temperature_c": 22.0,
                "start_time": "2026-01-01T09:00:00",
            },
            "retrieved_docs": [{"source": "Book.pdf", "text": "rye", "score": 0.9}],
            "messages": [],
            "user_query": "plan rye bake",
        }

        result = build_timeline(state)
        recipe = result["bake_plan_data"]["recipe"]
        flour_g = recipe["flour_g"]

        # Should fall back to country_loaf defaults: 20% starter, 2% salt
        assert recipe["starter_g"] == pytest.approx(flour_g * 0.20, abs=2.0)
        assert recipe["salt_g"] == pytest.approx(flour_g * 0.02, abs=1.0)
