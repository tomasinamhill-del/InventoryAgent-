"""
Unit tests for deterministic inventory tools and decision agent.
Tests:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (all 4 variability cases)
4. calculate_periodic_review()
5. calculate_newsvendor()
And agent pipeline integration tests.
"""

import math
import unittest

from inventory_tools import (
    normal_standard_pdf,
    normal_standard_cdf,
    normal_standard_inv_cdf,
    normal_loss_function,
    calculate_eoq,
    EOQInputs,
    EOQResult,
    calculate_quantity_discount,
    QuantityDiscountInputs,
    DiscountTier,
    QuantityDiscountResult,
    calculate_rop,
    SafetyStockROPInputs,
    ROPResult,
    calculate_periodic_review,
    FixedPeriodReviewInputs,
    PeriodicReviewResult,
    calculate_newsvendor,
    NewsvendorInputs,
    NewsvendorResult,
    AnalystVerificationAudit,
)
from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
)


class TestStatisticalHelpers(unittest.TestCase):
    def test_standard_normal_cdf_and_inv(self):
        # 50% percentile is 0.0
        self.assertAlmostEqual(normal_standard_cdf(0.0), 0.5, places=5)
        self.assertAlmostEqual(normal_standard_inv_cdf(0.5), 0.0, places=5)

        # 95% service level z is approx 1.64485
        z_95 = normal_standard_inv_cdf(0.95)
        self.assertAlmostEqual(z_95, 1.64485, places=4)
        self.assertAlmostEqual(normal_standard_cdf(z_95), 0.95, places=5)

        # 99% service level z is approx 2.32635
        z_99 = normal_standard_inv_cdf(0.99)
        self.assertAlmostEqual(z_99, 2.32635, places=4)

    def test_normal_loss_function(self):
        l_zero = normal_loss_function(0.0)
        self.assertAlmostEqual(l_zero, 1.0 / math.sqrt(2 * math.pi), places=5)
        l_high = normal_loss_function(3.0)
        self.assertLess(l_high, 0.005)
        self.assertGreater(l_high, 0.0)


# ============================================================================
# 1. Tests for calculate_eoq()
# ============================================================================

class TestCalculateEOQ(unittest.TestCase):
    def test_eoq_textbook_calculation(self):
        # D = 12,000, S = $50, H = $2/unit/year
        res = calculate_eoq(annual_demand=12000.0, order_cost=50.0, holding_cost=2.0)

        # Q = sqrt(2 * 12000 * 50 / 2) = sqrt(600,000) = 774.596669...
        expected_q = math.sqrt(600000.0)
        self.assertAlmostEqual(res.final_answer, expected_q, places=4)
        self.assertAlmostEqual(res.eoq, expected_q, places=4)

        # Verify intermediate calculations
        self.assertIn("intermediate_calculations", res.to_dict())
        self.assertAlmostEqual(res.annual_ordering_cost, res.annual_holding_cost, places=4)
        self.assertAlmostEqual(res.orders_per_year, 12000.0 / expected_q, places=4)
        self.assertAlmostEqual(res.total_annual_inventory_cost, 2.0 * expected_q, places=3)
        self.assertTrue(res.intermediate_calculations["is_global_convex_minimum"])

        # Verify assumptions
        self.assertGreater(len(res.assumptions), 0)
        self.assertTrue(any("constant" in a.lower() for a in res.assumptions))

    def test_eoq_input_validation(self):
        # Missing inputs must raise ValueError (never invent missing numerical inputs)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=None, order_cost=50.0, holding_cost=2.0)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=1000.0, order_cost=None, holding_cost=2.0)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=1000.0, order_cost=50.0, holding_cost=None)

        # Non-positive numbers must raise ValueError
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=0, order_cost=50, holding_cost=2)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=1000, order_cost=-10, holding_cost=2)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=1000, order_cost=50, holding_cost=0)

    def test_eoq_with_unit_cost_and_lead_time(self):
        res = calculate_eoq(
            annual_demand=12000.0,
            order_cost=50.0,
            holding_cost=2.0,
            unit_cost=10.0,
            lead_time_days=5.0,
            work_days_per_year=250,
        )
        self.assertEqual(res.annual_purchase_cost, 120000.0)
        self.assertEqual(res.reorder_point, 240.0)  # (12000/250) * 5


# ============================================================================
# 2. Tests for calculate_quantity_discount()
# ============================================================================

class TestCalculateQuantityDiscount(unittest.TestCase):
    def setUp(self):
        self.tiers = [
            DiscountTier(min_qty=1, max_qty=499, unit_price=10.00, tier_name="Tier 1"),
            DiscountTier(min_qty=500, max_qty=999, unit_price=9.50, tier_name="Tier 2"),
            DiscountTier(min_qty=1000, max_qty=None, unit_price=9.00, tier_name="Tier 3"),
        ]

    def test_percentage_holding_cost_evaluation(self):
        # D = 4,000, S = $45, holding rate = 20%
        res = calculate_quantity_discount(
            annual_demand=4000.0,
            order_cost=45.0,
            tiers=self.tiers,
            holding_rate=0.20,
        )

        # Tier 3 (1000+) should minimize total annual cost
        self.assertEqual(res.optimal_tier_index, 2)
        self.assertEqual(res.final_answer, 1000.0)
        self.assertEqual(res.optimal_order_quantity, 1000.0)
        self.assertAlmostEqual(res.optimal_total_annual_cost, 37080.0, places=2)
        self.assertGreater(res.annual_savings_vs_base, 3700.0)

        # Check intermediate calculations & assumptions
        self.assertEqual(len(res.tier_evaluations), 3)
        self.assertIn("evaluated_tiers", res.intermediate_calculations)
        self.assertGreater(len(res.assumptions), 0)

    def test_fixed_holding_cost_evaluation(self):
        res = calculate_quantity_discount(
            annual_demand=5000.0,
            order_cost=60.0,
            tiers=[
                DiscountTier(min_qty=1, max_qty=999, unit_price=15.00),
                DiscountTier(min_qty=1000, max_qty=None, unit_price=14.00),
            ],
            fixed_holding_cost=3.00,
        )
        self.assertIsNotNone(res.final_answer)
        self.assertGreater(res.final_answer, 0)
        self.assertEqual(res.intermediate_calculations["holding_cost_mechanism"], "fixed_dollar_amount")

    def test_quantity_discount_validation(self):
        # Missing holding mechanism must fail
        with self.assertRaises(ValueError):
            calculate_quantity_discount(annual_demand=4000.0, order_cost=45.0, tiers=self.tiers)

        # Ambiguous holding mechanism must fail
        with self.assertRaises(ValueError):
            calculate_quantity_discount(
                annual_demand=4000.0, order_cost=45.0, tiers=self.tiers,
                holding_rate=0.2, fixed_holding_cost=2.0
            )

        # Empty tiers must fail
        with self.assertRaises(ValueError):
            calculate_quantity_discount(annual_demand=4000.0, order_cost=45.0, tiers=[], holding_rate=0.2)


# ============================================================================
# 3. Tests for calculate_rop() (All 4 Cases)
# ============================================================================

class TestCalculateROP(unittest.TestCase):
    def test_case_1_constant_demand_and_constant_lead_time(self):
        # d = 50, L = 4, sigma_d = 0, sigma_L = 0
        res = calculate_rop(
            demand_mean=50.0,
            lead_time_mean=4.0,
            demand_std=0.0,
            lead_time_std=0.0,
        )
        self.assertEqual(res.case_type, "constant_demand_and_constant_lead_time")
        self.assertEqual(res.safety_stock, 0.0)
        self.assertEqual(res.final_answer, 200.0)  # d * L = 50 * 4 = 200
        self.assertEqual(res.reorder_point, 200.0)
        self.assertTrue(any("Case 1" in a for a in res.assumptions))

    def test_case_2_variable_demand_and_constant_lead_time(self):
        # d = 100, L = 4, sigma_d = 15, sigma_L = 0, CSL = 95%
        # sigma_LT = 15 * sqrt(4) = 30
        # z for 95% is ~1.64485
        # SS = 1.64485 * 30 = 49.3456
        # ROP = 400 + 49.3456 = 449.3456
        res = calculate_rop(
            demand_mean=100.0,
            lead_time_mean=4.0,
            demand_std=15.0,
            lead_time_std=0.0,
            service_level=0.95,
        )
        self.assertEqual(res.case_type, "variable_demand_and_constant_lead_time")
        self.assertAlmostEqual(res.lead_time_demand_std, 30.0, places=3)
        self.assertAlmostEqual(res.safety_stock, 49.3456, places=2)
        self.assertAlmostEqual(res.final_answer, 449.3456, places=2)
        self.assertTrue(any("Case 2" in a for a in res.assumptions))

    def test_case_3_constant_demand_and_variable_lead_time(self):
        # d = 80, L = 5, sigma_d = 0, sigma_L = 1.5, z = 2.0 (approx 97.7% CSL)
        # sigma_LT = d * sigma_L = 80 * 1.5 = 120.0
        # SS = 2.0 * 120.0 = 240.0
        # ROP = 80 * 5 + 240 = 400 + 240 = 640.0
        res = calculate_rop(
            demand_mean=80.0,
            lead_time_mean=5.0,
            demand_std=0.0,
            lead_time_std=1.5,
            z_score=2.0,
        )
        self.assertEqual(res.case_type, "constant_demand_and_variable_lead_time")
        self.assertAlmostEqual(res.lead_time_demand_std, 120.0, places=3)
        self.assertAlmostEqual(res.safety_stock, 240.0, places=2)
        self.assertAlmostEqual(res.final_answer, 640.0, places=2)
        self.assertTrue(any("Case 3" in a for a in res.assumptions))

    def test_case_4_variable_demand_and_variable_lead_time(self):
        # d = 100, L = 4, sigma_d = 15, sigma_L = 1.0, CSL = 95%
        # sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        # variance = 4 * 225 + 10000 * 1 = 900 + 10000 = 10900
        # sigma_LT = sqrt(10900) approx 104.403065
        # SS = 1.64485 * 104.403065 approx 171.728
        # ROP = 400 + 171.728 = 571.728
        res = calculate_rop(
            demand_mean=100.0,
            lead_time_mean=4.0,
            demand_std=15.0,
            lead_time_std=1.0,
            service_level=0.95,
        )
        self.assertEqual(res.case_type, "variable_demand_and_variable_lead_time")
        expected_sigma_lt = math.sqrt(10900.0)
        self.assertAlmostEqual(res.lead_time_demand_std, expected_sigma_lt, places=3)
        self.assertAlmostEqual(res.safety_stock, 1.64485 * expected_sigma_lt, places=1)
        self.assertAlmostEqual(res.final_answer, 400.0 + (1.64485 * expected_sigma_lt), places=1)
        self.assertTrue(any("Case 4" in a for a in res.assumptions))

    def test_rop_validation_missing_inputs(self):
        # Missing demand_mean
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=None, lead_time_mean=4.0)

        # Missing lead_time_mean
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=100.0, lead_time_mean=None)

        # Stochastic without service level or z_score
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=100.0, lead_time_mean=4.0, demand_std=15.0)

        # Invalid service level bounds
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=100.0, lead_time_mean=4.0, demand_std=15.0, service_level=1.5)


# ============================================================================
# 4. Tests for calculate_periodic_review()
# ============================================================================

class TestCalculatePeriodicReview(unittest.TestCase):
    def test_periodic_review_calculation(self):
        # R = 14, L = 7 -> Protection Period = 21
        # d = 50, sigma_d = 8, CSL = 98% (z approx 2.05375)
        # Demand during protection period = 50 * 21 = 1050
        # std during protection period = 8 * sqrt(21) approx 36.6606
        # SS = 2.05375 * 36.6606 approx 75.29
        # Target inventory S = 1050 + 75.29 = 1125.29
        # Inventory position = 520 (on_hand 420 + on_order 100)
        # Order quantity q = 1125.29 - 520 = 605.29
        res = calculate_periodic_review(
            review_period=14.0,
            lead_time=7.0,
            demand_mean_per_period=50.0,
            demand_std_per_period=8.0,
            service_level=0.98,
            inventory_position=520.0,
        )

        self.assertEqual(res.protection_period, 21.0)
        self.assertAlmostEqual(res.protection_demand_mean, 1050.0, places=2)
        self.assertAlmostEqual(res.protection_demand_std, 8.0 * math.sqrt(21.0), places=3)
        self.assertAlmostEqual(res.target_inventory, 1125.29, places=1)
        self.assertAlmostEqual(res.final_answer, 605.29, places=1)
        self.assertAlmostEqual(res.recommended_order_quantity, 605.29, places=1)

        # Verify assumptions and intermediate calculations
        self.assertIn("protection_period_R_plus_L", res.intermediate_calculations)
        self.assertGreater(len(res.assumptions), 0)

    def test_periodic_review_non_negative_order_clipping(self):
        # If inventory position exceeds target inventory, order quantity is max(0, S - IP) = 0
        res = calculate_periodic_review(
            review_period=7.0,
            lead_time=2.0,
            demand_mean_per_period=10.0,
            demand_std_per_period=0.0,
            inventory_position=100.0,  # S = 10 * 9 = 90
        )
        self.assertEqual(res.target_inventory, 90.0)
        self.assertEqual(res.final_answer, 0.0)

    def test_periodic_review_validation(self):
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=0, lead_time=5, demand_mean_per_period=10)
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=7, lead_time=-1, demand_mean_per_period=10)


# ============================================================================
# 5. Tests for calculate_newsvendor()
# ============================================================================

class TestCalculateNewsvendor(unittest.TestCase):
    def test_newsvendor_calculation(self):
        # P = 250, C = 100, V = 40, g = 0
        # Cu = 250 - 100 = 150
        # Co = 100 - 40 = 60
        # CR = 150 / (150 + 60) = 150 / 210 = 0.7142857
        # mu = 500, sigma = 80
        # z for 0.7142857 is approx 0.5659
        # Q* = 500 + 0.5659 * 80 = 545.28
        res = calculate_newsvendor(
            selling_price=250.0,
            unit_cost=100.0,
            demand_mean=500.0,
            demand_std=80.0,
            salvage_value=40.0,
        )

        self.assertEqual(res.underage_cost, 150.0)
        self.assertEqual(res.overage_cost, 60.0)
        self.assertAlmostEqual(res.critical_ratio, 150.0 / 210.0, places=5)
        self.assertAlmostEqual(res.final_answer, 545.28, places=1)
        self.assertAlmostEqual(res.recommended_order_quantity, 545.28, places=1)

        # Verify intermediate calculations & assumptions
        self.assertIn("expected_profit", res.intermediate_calculations)
        self.assertGreater(res.expected_profit, 0)
        self.assertGreater(len(res.assumptions), 0)

    def test_newsvendor_with_stockout_penalty(self):
        # P = 100, C = 60, V = 20, g = 10
        # Cu = 100 - 60 + 10 = 50
        # Co = 60 - 20 = 40
        # CR = 50 / 90 = 0.5555...
        res = calculate_newsvendor(
            selling_price=100.0,
            unit_cost=60.0,
            demand_mean=200.0,
            demand_std=30.0,
            salvage_value=20.0,
            stockout_penalty=10.0,
        )
        self.assertEqual(res.underage_cost, 50.0)
        self.assertEqual(res.overage_cost, 40.0)
        self.assertAlmostEqual(res.critical_ratio, 50.0 / 90.0, places=5)
        self.assertGreater(res.final_answer, 200.0)

    def test_newsvendor_validation(self):
        # Price <= cost must fail
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=50, unit_cost=60, demand_mean=100, demand_std=10)

        # Salvage >= cost must fail
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=100, unit_cost=50, salvage_value=60, demand_mean=100, demand_std=10)


# ============================================================================
# 6. Tests for InventoryDecisionAgent End-to-End
# ============================================================================

class TestInventoryDecisionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = InventoryDecisionAgent()

    def test_agent_eoq_query(self):
        q = "Our annual demand is 12,000 units. Each order costs $50 to place, and holding cost is $2 per unit per year. What is our EOQ?"
        res = self.agent.process_query(q)

        self.assertEqual(res.identified_model, InventoryModelType.EOQ)
        self.assertEqual(res.tool_selected, "inventory_tools.calculate_eoq")
        self.assertTrue(res.is_verified)
        self.assertAlmostEqual(res.calculation_result.final_answer, math.sqrt(600000), places=3)

    def test_agent_quantity_discount_query(self):
        q = "A supplier offers volume discount: 1 to 499 units @ $10.00, 500 to 999 units @ $9.50, and 1000+ units @ $9.00. Annual demand is 4,000 units, order cost is $45, and annual holding rate is 20%. How much should we order?"
        res = self.agent.process_query(q)

        self.assertEqual(res.identified_model, InventoryModelType.QUANTITY_DISCOUNT)
        self.assertEqual(res.tool_selected, "inventory_tools.calculate_quantity_discount")
        self.assertEqual(res.calculation_result.final_answer, 1000.0)

    def test_agent_safety_stock_rop_query(self):
        q = "Daily demand averages 100 units with standard deviation of 15 units. Supplier lead time is 4 days. We need a 95% cycle service level. What should our safety stock and reorder point be?"
        res = self.agent.process_query(q)

        self.assertEqual(res.identified_model, InventoryModelType.SAFETY_STOCK_ROP)
        self.assertEqual(res.tool_selected, "inventory_tools.calculate_rop")
        self.assertAlmostEqual(res.calculation_result.final_answer, 449.3456, places=2)

    def test_agent_fixed_period_review_query(self):
        q = "We review inventory every 14 days. Lead time is 7 days. Average demand is 50 units with std dev 8. Target service level is 98%. On hand we have 420 units and 100 on order. What is the order-up-to level and how much should we order?"
        res = self.agent.process_query(q)

        self.assertEqual(res.identified_model, InventoryModelType.FIXED_PERIOD_REVIEW)
        self.assertEqual(res.tool_selected, "inventory_tools.calculate_periodic_review")
        self.assertAlmostEqual(res.calculation_result.final_answer, 605.29, places=1)

    def test_agent_newsvendor_query(self):
        q = "We sell winter jackets for $250 each. We buy them for $100. Any unsold jackets at the end of the season are sold at markdown for $40. Demand is normally distributed with mean 500 and std dev 80. How many jackets should we purchase?"
        res = self.agent.process_query(q)

        self.assertEqual(res.identified_model, InventoryModelType.NEWSVENDOR)
        self.assertEqual(res.tool_selected, "inventory_tools.calculate_newsvendor")
        self.assertAlmostEqual(res.calculation_result.critical_ratio, 150.0 / 210.0, places=4)
        self.assertAlmostEqual(res.calculation_result.final_answer, 545.28, places=1)


class TestNaturalLanguageInputExtraction(unittest.TestCase):
    """
    Exhaustive tests for natural-language extraction:
    - 7 required ordering cost synonyms for EOQ
    - Parameter distinctions for Safety Stock / ROP
    - Parameter distinctions for Quantity Discount
    - Parameter distinctions for Fixed-Period Review (including IP = OH + OO - B)
    - Parameter distinctions for Newsvendor (including direct Cu / Co)
    """

    def setUp(self):
        self.agent = InventoryDecisionAgent()

    def test_eoq_all_7_ordering_cost_synonyms(self):
        """Verify all 7 required ordering cost synonyms are accurately recognized."""
        synonyms = [
            ("ordering cost", "Annual demand is 10000 units. The ordering cost is $50. Holding cost is $2 per unit per year."),
            ("order cost", "Annual demand is 10000 units. The order cost is $50. Holding cost is $2 per unit per year."),
            ("setup cost", "Annual demand is 10000 units. The setup cost is $50. Holding cost is $2 per unit per year."),
            ("setup expense", "Annual demand is 10000 units. The setup expense is $50. Holding cost is $2 per unit per year."),
            ("cost per order", "Annual demand is 10000 units. The cost per order is $50. Holding cost is $2 per unit per year."),
            ("cost of placing an order", "Annual demand is 10000 units. The cost of placing an order is $50. Holding cost is $2 per unit per year."),
            ("ordering/setup cost", "Annual demand is 10000 units. The ordering/setup cost is $50. Holding cost is $2 per unit per year."),
        ]

        for syn_name, query in synonyms:
            with self.subTest(synonym=syn_name):
                inputs, missing = self.agent.extract_inputs(query, InventoryModelType.EOQ)
                self.assertEqual(missing, [], f"Synonym '{syn_name}' failed with missing fields: {missing}")
                self.assertEqual(inputs.get("order_cost"), 50.0, f"Failed to extract order_cost for synonym '{syn_name}'")
                self.assertEqual(inputs.get("annual_demand"), 10000.0)
                self.assertEqual(inputs.get("holding_cost"), 2.0)

    def test_safety_stock_rop_parameter_distinction(self):
        """Distinguish average demand, demand std dev, average lead time, lead-time std dev, service level."""
        # Stochastic demand and stochastic lead time
        q_stochastic = (
            "Average demand is 120 units per day with a demand standard deviation of 18 units. "
            "Average lead time is 5 days with a lead-time standard deviation of 1.2 days. "
            "Target cycle service level is 95%."
        )
        inputs, missing = self.agent.extract_inputs(q_stochastic, InventoryModelType.SAFETY_STOCK_ROP)
        self.assertEqual(missing, [])
        self.assertEqual(inputs["avg_demand_per_period"], 120.0)
        self.assertEqual(inputs["std_demand_per_period"], 18.0)
        self.assertEqual(inputs["lead_time_periods"], 5.0)
        self.assertEqual(inputs["std_lead_time_periods"], 1.2)
        self.assertEqual(inputs["service_level"], 0.95)

        # Deterministic / constant demand case: std dev must NOT be declared missing
        q_constant = "Daily demand is constant at 100 units per day. Supplier lead time is 4 days. What is ROP?"
        inputs_c, missing_c = self.agent.extract_inputs(q_constant, InventoryModelType.SAFETY_STOCK_ROP)
        self.assertEqual(missing_c, [])
        self.assertEqual(inputs_c["avg_demand_per_period"], 100.0)
        self.assertEqual(inputs_c["std_demand_per_period"], 0.0)
        self.assertEqual(inputs_c["lead_time_periods"], 4.0)

    def test_quantity_discount_parameter_distinction(self):
        """Distinguish quantity break, unit price, ordering cost, fixed vs percentage holding cost."""
        # With percentage holding cost rate
        q_pct = (
            "Annual demand is 5000 units. Setup cost is $60. Carrying rate is 15% of unit price. "
            "Tiers: 1 to 499 units at $20.00, 500 to 999 units at $18.50, 1000+ units at $17.00."
        )
        inputs, missing = self.agent.extract_inputs(q_pct, InventoryModelType.QUANTITY_DISCOUNT)
        self.assertEqual(missing, [])
        self.assertEqual(inputs["annual_demand"], 5000.0)
        self.assertEqual(inputs["order_cost"], 60.0)
        self.assertEqual(inputs["holding_rate"], 0.15)
        self.assertNotIn("fixed_holding_cost", inputs)
        self.assertEqual(len(inputs["tiers"]), 3)
        self.assertEqual(inputs["tiers"][0].unit_price, 20.0)
        self.assertEqual(inputs["tiers"][1].min_qty, 500.0)
        self.assertEqual(inputs["tiers"][2].unit_price, 17.0)

        # With fixed holding cost
        q_fixed = (
            "Annual demand is 8000 units. Ordering cost is $40. Fixed holding cost is $3.50 per unit. "
            "Price breaks: 1-499: $12.00, 500-999: $11.00, 1000+: $10.00."
        )
        inputs_f, missing_f = self.agent.extract_inputs(q_fixed, InventoryModelType.QUANTITY_DISCOUNT)
        self.assertEqual(missing_f, [])
        self.assertEqual(inputs_f["annual_demand"], 8000.0)
        self.assertEqual(inputs_f["order_cost"], 40.0)
        self.assertEqual(inputs_f["fixed_holding_cost"], 3.50)
        self.assertNotIn("holding_rate", inputs_f)

    def test_fixed_period_review_parameter_distinction(self):
        """Distinguish demand, review period, lead time, safety stock, inventory position, on-hand, on-order, backorders."""
        # 1. With On-hand, On-order, and Backorders: IP = OH + OO - B
        q1 = (
            "We review inventory every 7 days. Lead time is 3 days. Demand averages 40 units per day with std dev of 6. "
            "Service level is 95%. Currently we have 300 units on-hand, 150 units on-order, and 50 backorders."
        )
        inputs1, missing1 = self.agent.extract_inputs(q1, InventoryModelType.FIXED_PERIOD_REVIEW)
        self.assertEqual(missing1, [])
        self.assertEqual(inputs1["review_period"], 7.0)
        self.assertEqual(inputs1["lead_time"], 3.0)
        self.assertEqual(inputs1["avg_demand_per_period"], 40.0)
        self.assertEqual(inputs1["std_demand_per_period"], 6.0)
        self.assertEqual(inputs1["current_on_hand"], 300.0)
        self.assertEqual(inputs1["current_on_order"], 150.0)
        self.assertEqual(inputs1["backorders"], 50.0)
        # IP = 300 + 150 - 50 = 400
        self.assertEqual(inputs1["inventory_position"], 400.0)

        # 2. With direct Safety Stock specified
        q2 = (
            "We review every 14 days with lead time of 7 days. Demand is 50 units per day. "
            "We keep a safety stock of 80 units. Inventory position is 350 units."
        )
        inputs2, missing2 = self.agent.extract_inputs(q2, InventoryModelType.FIXED_PERIOD_REVIEW)
        self.assertEqual(missing2, [])
        self.assertEqual(inputs2["safety_stock"], 80.0)
        self.assertEqual(inputs2["inventory_position"], 350.0)

    def test_newsvendor_parameter_distinction(self):
        """Distinguish selling price, purchase cost, salvage value, underage/overage costs, demand distribution."""
        # 1. Direct underage (Cu) and overage (Co) costs given
        q_direct = (
            "Underage cost is $15 per unit and overage cost is $5 per unit. "
            "Demand is normally distributed with mean 200 units and standard deviation 30 units."
        )
        inputs_d, missing_d = self.agent.extract_inputs(q_direct, InventoryModelType.NEWSVENDOR)
        # selling_price and purchase_cost should NOT be reported missing!
        self.assertEqual(missing_d, [])
        self.assertEqual(inputs_d["underage_cost"], 15.0)
        self.assertEqual(inputs_d["overage_cost"], 5.0)
        self.assertEqual(inputs_d["mean_demand"], 200.0)
        self.assertEqual(inputs_d["std_demand"], 30.0)

        # 2. Selling price, unit cost, salvage value, stockout penalty given
        q_full = (
            "We sell jackets for $200 each and buy them for $80 each. Unsold jackets have a salvage value of $30. "
            "Stockout penalty is $10. Mean demand is 600 units with standard deviation of 90."
        )
        inputs_f, missing_f = self.agent.extract_inputs(q_full, InventoryModelType.NEWSVENDOR)
        self.assertEqual(missing_f, [])
        self.assertEqual(inputs_f["selling_price"], 200.0)
        self.assertEqual(inputs_f["unit_cost"], 80.0)
        self.assertEqual(inputs_f["salvage_value"], 30.0)
        self.assertEqual(inputs_f["stockout_penalty"], 10.0)
        self.assertEqual(inputs_f["mean_demand"], 600.0)
        self.assertEqual(inputs_f["std_demand"], 90.0)
        self.assertEqual(inputs_f["demand_distribution"], "normal")



class TestAnalystVerificationAudit(unittest.TestCase):
    """
    Verify that every calculation tool produces a complete, structured
    Step-by-Step Analyst Verification Audit directly from Python, covering:
    1. Extracted Inputs & Units
    2. Variable Definitions
    3. Base Formula
    4. Formula with Substituted Values
    5. Step-by-Step Intermediate Python Calculations
    6. Final Verified Result
    """

    def setUp(self):
        self.agent = InventoryDecisionAgent()

    def _assert_valid_audit(self, audit: AnalystVerificationAudit, expected_model_name: str):
        self.assertIsInstance(audit, AnalystVerificationAudit)
        self.assertEqual(audit.model_name, expected_model_name)
        self.assertTrue(len(audit.extracted_inputs_and_units) > 0)
        self.assertTrue(len(audit.variable_definitions) > 0)
        self.assertTrue(len(audit.base_formula) > 0)
        self.assertTrue(len(audit.formula_with_substituted_values) > 0)
        self.assertTrue(len(audit.intermediate_calculations) >= 3)
        self.assertTrue(len(audit.final_verified_result) > 0)

        # Markdown representation contains all 6 required section headers
        md = audit.to_markdown()
        self.assertIn("1. Extracted Inputs & Units", md)
        self.assertIn("2. Mathematical Variable Definitions", md)
        self.assertIn("3. Base Mathematical Formulas", md)
        self.assertIn("4. Substituted Mathematical Formulas", md)
        self.assertIn("5. Step-by-Step Intermediate Python Calculations", md)
        self.assertIn("6. Final Verified Result & Invariant Checks", md)

    def test_eoq_audit_generation(self):
        res = calculate_eoq(EOQInputs(annual_demand=12000, order_cost=50, holding_cost=2.0))
        self.assertIsNotNone(res.audit)
        self.assertIsNotNone(res.analyst_verification_audit)
        self._assert_valid_audit(res.audit, "Economic Order Quantity (EOQ)")

    def test_quantity_discount_audit_generation(self):
        tiers = [
            DiscountTier(min_qty=1, max_qty=499, unit_price=10.0, tier_name="Tier 1"),
            DiscountTier(min_qty=500, max_qty=999, unit_price=9.5, tier_name="Tier 2"),
            DiscountTier(min_qty=1000, max_qty=None, unit_price=9.0, tier_name="Tier 3"),
        ]
        res = calculate_quantity_discount(QuantityDiscountInputs(annual_demand=4000, order_cost=45, holding_rate=0.2, tiers=tiers))
        self.assertIsNotNone(res.audit)
        self._assert_valid_audit(res.audit, "All-Units Quantity Discount Model")

    def test_rop_audit_generation(self):
        res = calculate_rop(SafetyStockROPInputs(
            avg_demand_per_period=100,
            std_demand_per_period=15,
            lead_time_periods=4,
            service_level=0.95,
        ))
        self.assertIsNotNone(res.audit)
        self._assert_valid_audit(res.audit, "Safety Stock and Reorder Point (Continuous Review)")

    def test_periodic_review_audit_generation(self):
        res = calculate_periodic_review(FixedPeriodReviewInputs(
            review_period=14,
            lead_time=7,
            avg_demand_per_period=50,
            std_demand_per_period=8,
            service_level=0.98,
            current_on_hand=420,
            current_on_order=100,
        ))
        self.assertIsNotNone(res.audit)
        self._assert_valid_audit(res.audit, "Fixed-Period Review Model (Periodic Review / Order-Up-To)")

    def test_newsvendor_audit_generation(self):
        res = calculate_newsvendor(NewsvendorInputs(
            selling_price=250,
            unit_cost=100,
            salvage_value=40,
            mean_demand=500,
            std_demand=80,
        ))
        self.assertIsNotNone(res.audit)
        self._assert_valid_audit(res.audit, "Newsvendor Model (Single-Period Stochastic)")

    def test_agent_workflow_attaches_audit(self):
        q = "Annual demand is 10000 units, ordering cost is $60, and holding cost is $3 per unit. What is the EOQ?"
        workflow_res = self.agent.process_query(q)
        self.assertIsNotNone(workflow_res.audit)
        self.assertIsNotNone(workflow_res.analyst_verification_audit)
        self._assert_valid_audit(workflow_res.audit, "Economic Order Quantity (EOQ)")


if __name__ == "__main__":
    unittest.main()
