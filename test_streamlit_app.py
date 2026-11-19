"""
Unit tests for streamlit_app module logic, data structures, and examples.
"""

from __future__ import annotations

import unittest
import re
from inventory_agent import InventoryDecisionAgent, InventoryModelType
from streamlit_app import (
    VARIABLE_DEFINITIONS,
    EXAMPLES,
    SAMPLE_INVENTORY_CSV,
    format_formula_for_display,
    build_verification_table,
    normalize_column_name,
    clean_numeric_series,
    load_and_process_inventory_csv,
    compute_inventory_metrics,
    compute_category_breakdown,
)


class TestStreamlitAppComponents(unittest.TestCase):
    def setUp(self):
        self.agent = InventoryDecisionAgent()

    def test_variable_definitions_completeness(self):
        """Ensure variable definitions exist for all 5 inventory models."""
        expected_models = [
            InventoryModelType.EOQ,
            InventoryModelType.QUANTITY_DISCOUNT,
            InventoryModelType.SAFETY_STOCK_ROP,
            InventoryModelType.FIXED_PERIOD_REVIEW,
            InventoryModelType.NEWSVENDOR,
        ]
        for m in expected_models:
            self.assertIn(m, VARIABLE_DEFINITIONS)
            defs = VARIABLE_DEFINITIONS[m]
            self.assertGreater(len(defs), 5, f"Too few definitions for {m.value}")
            for item in defs:
                self.assertIn("Symbol", item)
                self.assertIn("Name", item)
                self.assertIn("Definition", item)
                self.assertIn("Units", item)

    def test_all_example_problems_executable(self):
        """Verify that clicking any of the 5 example problem buttons produces a valid, verified solution."""
        self.assertEqual(len(EXAMPLES), 5)
        for ex in EXAMPLES:
            q = ex["question"]
            res = self.agent.process_query(q)
            self.assertEqual(res.missing_inputs, [], f"Missing inputs for example '{ex['label']}'")
            self.assertIsNone(res.error_message)
            self.assertIsNotNone(res.calculation_result)
            self.assertIsNotNone(res.explanation)
            self.assertTrue(res.is_verified)
            self.assertGreater(len(res.explanation.formula_trace), 0)
            self.assertGreater(len(res.explanation.managerial_insights), 0)
            self.assertIsNotNone(res.explanation.recommended_action)

    def test_format_formula_for_display_escapes_and_separates_currency(self):
        """Verify Section 3 formula formatting renders currency outside LaTeX $...$ blocks and escapes \\$."""
        # Test EOQ Steps 3, 4, 5 with raw currency dollar signs
        raw_step3 = r"3. **Annual Ordering Cost:** $\frac{D}{Q^*} S = 15.49 \times \$50.00 = \$774.60$."
        clean_step3 = format_formula_for_display(raw_step3)
        self.assertIn(r"$\frac{D}{Q^*} S = 15.49 \times$", clean_step3)
        self.assertIn(r"\$50.00 = \$774.60", clean_step3)
        # Ensure no dollar sign remains inside the $...$ math block
        match = re.search(r"\$(.*?)\$", clean_step3)
        self.assertIsNotNone(match)
        self.assertNotIn("$", match.group(1))

        raw_step5 = r"5. **Total Annual Inventory Cost:** $\text{AOC} + \text{AHC} = \$1,549.19$."
        clean_step5 = format_formula_for_display(raw_step5)
        self.assertIn(r"$\text{AOC} + \text{AHC} =$", clean_step5)
        self.assertIn(r"\$1,549.19", clean_step5)
        match5 = re.search(r"\$(.*?)\$", clean_step5)
        self.assertIsNotNone(match5)
        self.assertNotIn("$", match5.group(1))

        # Test unescaped currency outside LaTeX
        raw_tier = "Tier 1 (Tier 1 @ $10.00): Total Cost = $9,500.00."
        clean_tier = format_formula_for_display(raw_tier)
        self.assertIn(r"\$10.00", clean_tier)
        self.assertIn(r"\$9,500.00", clean_tier)

        # Test idempotency (formatting already-clean strings does not mutate them)
        self.assertEqual(clean_step3, format_formula_for_display(clean_step3))
        self.assertEqual(clean_step5, format_formula_for_display(clean_step5))
        self.assertEqual(clean_tier, format_formula_for_display(clean_tier))

    def test_build_verification_table_evaluates_booleans_and_formats_numerics(self):
        """Verify Section 4 only evaluates booleans as ✓ PASS / ✗ FAIL, and numbers as formatted values."""
        sample_ver = {
            "is_valid": True,
            "holding_equals_ordering": True,
            "cost_difference": 0.0,
            "relative_cost_difference": 0.0,
            "second_derivative_strictly_positive": True,
            "lead_time_variability_included": False,
            "safety_stock_ratio_vs_continuous": 1.7320508,
            "total_tiers_evaluated": 3,
            "sensitivity": {"ignored": 1.23},
        }

        table = build_verification_table(sample_ver)
        table_dict = {row["Verification Invariant"]: row["Status / Value"] for row in table}

        # Check boolean checks
        self.assertEqual(table_dict["is_valid"], "✓ PASS")
        self.assertEqual(table_dict["holding_equals_ordering"], "✓ PASS")
        self.assertEqual(table_dict["second_derivative_strictly_positive"], "✓ PASS")
        self.assertEqual(table_dict["lead_time_variability_included"], "✗ FAIL")

        # Check numeric difference fields (must NOT be ✗ FAIL!)
        self.assertEqual(table_dict["cost_difference"], "0.0000")
        self.assertNotEqual(table_dict["cost_difference"], "✗ FAIL")
        self.assertEqual(table_dict["relative_cost_difference"], "0.0000")
        self.assertNotEqual(table_dict["relative_cost_difference"], "✗ FAIL")

        # Check other numeric metrics
        self.assertEqual(table_dict["safety_stock_ratio_vs_continuous"], "1.7321")
        self.assertEqual(table_dict["total_tiers_evaluated"], "3")

        # Verify excluded keys
        self.assertNotIn("sensitivity", table_dict)


class TestCSVInventoryAnalyticsDashboard(unittest.TestCase):
    """
    Unit tests for CSV Inventory Data Upload & Analytics Dashboard.
    Validates:
    - Case-insensitive column normalization and synonym recognition
    - CSV loading, validation, dataset summary, and original table preservation
    - Calculated inventory_value column (units * unit_cost)
    - Key dashboard metrics: total units, average unit cost, product with highest units,
      highest unit cost, and total inventory value across all products.
    - Category-level breakdown grouping total units and total inventory value by category.
    """

    def test_normalize_column_name_case_insensitive_and_synonyms(self):
        # Product synonyms & case
        self.assertEqual(normalize_column_name("product"), "product")
        self.assertEqual(normalize_column_name("PRODUCT"), "product")
        self.assertEqual(normalize_column_name("Product Name"), "product")
        self.assertEqual(normalize_column_name("item"), "product")
        self.assertEqual(normalize_column_name("SKU"), "product")

        # Category synonyms & case
        self.assertEqual(normalize_column_name("category"), "category")
        self.assertEqual(normalize_column_name("CATEGORY"), "category")
        self.assertEqual(normalize_column_name("Department"), "category")
        self.assertEqual(normalize_column_name("dept"), "category")

        # Units synonyms & case
        self.assertEqual(normalize_column_name("units"), "units")
        self.assertEqual(normalize_column_name("UNITS"), "units")
        self.assertEqual(normalize_column_name("Quantity"), "units")
        self.assertEqual(normalize_column_name("qty"), "units")
        self.assertEqual(normalize_column_name("on_hand"), "units")

        # Unit cost synonyms & case
        self.assertEqual(normalize_column_name("unit_cost"), "unit_cost")
        self.assertEqual(normalize_column_name("UNIT_COST"), "unit_cost")
        self.assertEqual(normalize_column_name("Unit Cost"), "unit_cost")
        self.assertEqual(normalize_column_name("unit_price"), "unit_cost")
        self.assertEqual(normalize_column_name("price"), "unit_cost")
        self.assertEqual(normalize_column_name("cost"), "unit_cost")

    def test_load_and_process_inventory_csv_sample(self):
        processed_df, orig_df, summary = load_and_process_inventory_csv(SAMPLE_INVENTORY_CSV)

        # Dataset summary checks
        self.assertEqual(summary["num_rows"], 10)
        self.assertEqual(summary["num_columns"], 4)
        self.assertEqual(summary["column_names"], ["product", "category", "units", "unit_cost"])

        # Original dataframe preserved
        self.assertEqual(len(orig_df), 10)

        # Calculated inventory_value column checks
        self.assertIn("inventory_value", processed_df.columns)
        records = processed_df.to_dict("records")
        # Widget Alpha: 500 units * $12.50 = $6,250.00
        alpha_rec = next(r for r in records if r["product"] == "Widget Alpha")
        self.assertEqual(alpha_rec["units"], 500.0)
        self.assertEqual(alpha_rec["unit_cost"], 12.50)
        self.assertEqual(alpha_rec["inventory_value"], 6250.0)

        # Industrial Motor 5HP: 40 units * $450.00 = $18,000.00
        motor_rec = next(r for r in records if r["product"] == "Industrial Motor 5HP")
        self.assertEqual(motor_rec["units"], 40.0)
        self.assertEqual(motor_rec["unit_cost"], 450.0)
        self.assertEqual(motor_rec["inventory_value"], 18000.0)

    def test_load_and_process_inventory_csv_case_variations_and_extra_columns(self):
        custom_csv = (
            "Item_Name,DEPT,Quantity,Unit Price,Lead_Time_Days\n"
            "Bolt Pack,Hardware,100,$2.50,5\n"
            "Bearing,Machinery,50,$20.00,10\n"
        )
        processed_df, orig_df, summary = load_and_process_inventory_csv(custom_csv)

        self.assertEqual(summary["num_rows"], 2)
        self.assertEqual(summary["num_columns"], 5)
        self.assertIn("inventory_value", processed_df.columns)
        self.assertIn("Lead_Time_Days", processed_df.columns)

        records = processed_df.to_dict("records")
        self.assertEqual(records[0]["inventory_value"], 250.0)
        self.assertEqual(records[1]["inventory_value"], 1000.0)

    def test_clean_numeric_series_currency_commas_whitespace(self):
        """Verify currency symbols ($), commas (,), and whitespace are stripped from numeric strings."""
        raw_values = ["$2.50", " $1,250.00 ", "1,000", "$0.45", " 25,500.50 ", 100, 15.5]
        cleaned = clean_numeric_series(raw_values)
        expected = ["2.50", "1250.00", "1000", "0.45", "25500.50", 100, 15.5]
        self.assertEqual(list(cleaned), expected)

    def test_load_and_process_inventory_csv_currency_formatted_prices_and_commas(self):
        """
        Verify that CSV files with currency-formatted prices ($2.50, $1,250.00) and
        comma-formatted quantities (1,000) parse accurately into numeric values
        instead of coercing to 0.0.
        """
        formatted_csv = (
            "product,category,units,unit_cost\n"
            "Standard Widget,Electronics,\"1,000\",\"$2.50\"\n"
            "High-End Assembly,Machinery,\"250\",\"$1,250.00\"\n"
            "Micro Sensor,Electronics,\"5,000\",\"$0.75\"\n"
        )
        processed_df, orig_df, summary = load_and_process_inventory_csv(formatted_csv)
        records = processed_df.to_dict("records")

        # Widget: 1,000 units @ $2.50 -> 2,500.00 (NOT 0.0!)
        self.assertEqual(records[0]["units"], 1000.0)
        self.assertEqual(records[0]["unit_cost"], 2.50)
        self.assertEqual(records[0]["inventory_value"], 2500.0)

        # High-End Assembly: 250 units @ $1,250.00 -> 312,500.00 (NOT 0.0!)
        self.assertEqual(records[1]["units"], 250.0)
        self.assertEqual(records[1]["unit_cost"], 1250.00)
        self.assertEqual(records[1]["inventory_value"], 312500.0)

        # Micro Sensor: 5,000 units @ $0.75 -> 3,750.00 (NOT 0.0!)
        self.assertEqual(records[2]["units"], 5000.0)
        self.assertEqual(records[2]["unit_cost"], 0.75)
        self.assertEqual(records[2]["inventory_value"], 3750.0)

        # Key metrics verification
        metrics = compute_inventory_metrics(processed_df)
        self.assertEqual(metrics["total_units"], 6250.0)
        self.assertEqual(metrics["highest_unit_cost"], 1250.00)
        self.assertEqual(metrics["product_highest_cost"], "High-End Assembly")
        self.assertEqual(metrics["total_inventory_value"], 318750.0)

    def test_load_and_process_inventory_csv_missing_columns_raises_error(self):
        bad_csv = "product,units,unit_cost\nWidget,100,5.00\n"
        with self.assertRaises(ValueError) as ctx:
            load_and_process_inventory_csv(bad_csv)
        self.assertIn("Missing required inventory column", str(ctx.exception))
        self.assertIn("category", str(ctx.exception))

    def test_compute_inventory_metrics(self):
        processed_df, _, _ = load_and_process_inventory_csv(SAMPLE_INVENTORY_CSV)
        metrics = compute_inventory_metrics(processed_df)

        # Key dashboard metrics
        self.assertEqual(metrics["total_units"], 8765.0)
        self.assertAlmostEqual(metrics["average_unit_cost"], 86.62, places=2)
        self.assertEqual(metrics["product_highest_units"], "Brass Hex Nut Pack")
        self.assertEqual(metrics["highest_units_count"], 3000.0)
        self.assertEqual(metrics["highest_unit_cost"], 450.0)
        self.assertEqual(metrics["product_highest_cost"], "Industrial Motor 5HP")
        self.assertAlmostEqual(metrics["total_inventory_value"], 84165.0, places=2)

    def test_compute_category_breakdown(self):
        processed_df, _, _ = load_and_process_inventory_csv(SAMPLE_INVENTORY_CSV)
        cat_breakdown = compute_category_breakdown(processed_df)

        self.assertEqual(len(cat_breakdown), 4)

        # Verified order sorted descending by total inventory value
        cat_names = [c["category"] for c in cat_breakdown]
        self.assertEqual(cat_names, ["Electronics", "Machinery", "Hardware", "Fasteners"])

        by_name = {c["category"]: c for c in cat_breakdown}

        # Electronics: 1,150 units, $39,200.00
        elec = by_name["Electronics"]
        self.assertEqual(elec["product_count"], 4)
        self.assertEqual(elec["total_units"], 1150.0)
        self.assertEqual(elec["total_inventory_value"], 39200.0)
        self.assertAlmostEqual(elec["pct_of_total_value"], (39200.0 / 84165.0) * 100.0, places=2)

        # Machinery: 115 units, $34,500.00
        mach = by_name["Machinery"]
        self.assertEqual(mach["product_count"], 2)
        self.assertEqual(mach["total_units"], 115.0)
        self.assertEqual(mach["total_inventory_value"], 34500.0)

        # Hardware: 2,000 units, $8,440.00
        hard = by_name["Hardware"]
        self.assertEqual(hard["total_units"], 2000.0)
        self.assertEqual(hard["total_inventory_value"], 8440.0)

        # Fasteners: 5,500 units, $2,025.00
        fast = by_name["Fasteners"]
        self.assertEqual(fast["total_units"], 5500.0)
        self.assertEqual(fast["total_inventory_value"], 2025.0)

        # Percentage sum checks
        total_pct_units = sum(c["pct_of_total_units"] for c in cat_breakdown)
        total_pct_val = sum(c["pct_of_total_value"] for c in cat_breakdown)
        self.assertAlmostEqual(total_pct_units, 100.0, places=2)
        self.assertAlmostEqual(total_pct_val, 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
