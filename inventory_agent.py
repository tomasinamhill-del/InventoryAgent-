"""
Supply Chain Analytics Inventory Decision Agent.
Orchestrates the 6-stage architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Dict, Any, List, Optional, Tuple

from inventory_tools import (
    calculate_eoq,
    EOQInputs,
    EOQResult,
    calculate_quantity_discount,
    QuantityDiscountInputs,
    DiscountTier,
    QuantityDiscountResult,
    calculate_rop,
    calculate_safety_stock_rop,
    SafetyStockROPInputs,
    SafetyStockROPResult,
    calculate_periodic_review,
    calculate_fixed_period_review,
    FixedPeriodReviewInputs,
    FixedPeriodReviewResult,
    calculate_newsvendor,
    NewsvendorInputs,
    NewsvendorResult,
    AnalystVerificationAudit,
)


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review (Periodic Review / Order-Up-To)"
    NEWSVENDOR = "Newsvendor Model (Single-Period Stochastic)"
    UNKNOWN = "Unknown / Clarification Needed"


@dataclass
class AgentExplanation:
    executive_summary: str
    formula_trace: List[str]
    managerial_insights: List[str]
    verification_summary: str
    recommended_action: str


@dataclass
class AgentWorkflowResult:
    customer_question: str
    identified_model: InventoryModelType
    model_confidence: float
    model_rationale: str
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    tool_selected: str
    calculation_result: Optional[Any]
    is_verified: bool
    verification_details: Dict[str, Any]
    explanation: Optional[AgentExplanation]
    error_message: Optional[str] = None
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Accepts natural-language inventory questions and executes the end-to-end pipeline:
    Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
    """

    def __init__(self):
        pass

    # ========================================================================
    # STAGE 1 & 2: Identify Model from Question
    # ========================================================================

    def identify_model(self, question: str) -> Tuple[InventoryModelType, float, str]:
        """
        Analyzes the natural language question and determines the appropriate inventory model.
        Returns: (model_type, confidence_score, rationale)
        """
        text = question.lower()

        scores: Dict[InventoryModelType, float] = {
            InventoryModelType.EOQ: 0.0,
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.NEWSVENDOR: 0.0,
        }

        # Model 2: Quantity Discount keywords
        qd_keywords = [
            "discount", "price break", "volume discount", "bulk discount", "tiers",
            "tier", "schedule", "all-units", "bracket", "pricing tier"
        ]
        for kw in qd_keywords:
            if kw in text:
                scores[InventoryModelType.QUANTITY_DISCOUNT] += 3.0

        # Look for multiple price or tier patterns e.g. "$10 for ... $9 for"
        if re.search(r"(\$\s*\d+.*){2,}", text) or re.search(r"\d+\s*[-–to]+\s*\d+", text):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 3.5

        # Model 5: Newsvendor keywords
        nv_keywords = [
            "newsvendor", "newsboy", "single period", "single-period", "perishable",
            "salvage", "markdown", "underage", "overage", "critical fractile",
            "critical ratio", "liquidation", "seasonal", "jackets", "newspaper",
            "perish", "goodwill", "unmet demand penalty"
        ]
        for kw in nv_keywords:
            if kw in text:
                scores[InventoryModelType.NEWSVENDOR] += 3.0

        if "salvage" in text or "markdown" in text or "liquidation" in text:
            scores[InventoryModelType.NEWSVENDOR] += 4.0

        # Model 4: Fixed-Period Review keywords
        fp_keywords = [
            "review period", "periodic review", "fixed period", "fixed-period",
            "order-up-to", "order up to", "base stock", "review every",
            "periodic inventory", "review interval", "on-order", "on order",
            "inventory position", "on-hand", "on hand"
        ]
        for kw in fp_keywords:
            if kw in text:
                scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3.0

        if re.search(r"review.*every\s+\d+", text) or re.search(r"every\s+\d+\s+(?:days?|weeks?|months?).*review", text):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 6.0
        elif re.search(r"review\s+(every|interval|period)", text):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 4.0

        if "review" in text and any(k in text for k in ["on-order", "on order", "backorders", "backorder", "inventory position", "order-up-to", "order up to"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 5.0

        # Model 1: EOQ keywords & structural indicators
        eoq_keywords = [
            "eoq", "economic order quantity", "annual demand", "holding cost",
            "order cost", "ordering cost", "setup cost", "optimal order quantity",
            "how much should we order", "how much to order", "wilson", "cycle inventory",
            "carrying cost"
        ]
        for kw in eoq_keywords:
            if kw in text:
                scores[InventoryModelType.EOQ] += 1.5

        has_holding = "holding" in text or "carrying" in text
        has_ordering = any(k in text for k in ["order cost", "ordering cost", "setup cost", "to place an order", "per order", "cost to place"])
        if has_holding and has_ordering:
            scores[InventoryModelType.EOQ] += 6.0

        if "optimal order quantity" in text or "economic order quantity" in text or "eoq" in text:
            scores[InventoryModelType.EOQ] += 6.0

        # Model 3: Safety Stock & Reorder Point keywords
        ss_keywords = [
            "safety stock", "cycle service level", "service level",
            "lead time demand", "stockout risk", "stockout probability", "continuous review",
            "standard deviation of lead time", "lead time variability"
        ]
        for kw in ss_keywords:
            if kw in text:
                scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.0

        if "safety stock" in text:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 6.0
        if "service level" in text or "cycle service level" in text or "csl" in text:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4.0
        if "reorder point" in text or "rop" in text:
            # If standard deviation or service level is present, it's stochastic ROP
            if any(k in text for k in ["standard deviation", "std dev", "service level", "csl", "safety stock"]):
                scores[InventoryModelType.SAFETY_STOCK_ROP] += 5.0
            else:
                # Deterministic ROP in EOQ
                scores[InventoryModelType.EOQ] += 2.0

        # Disambiguation between models:
        # If periodic review keywords are present, prefer FIXED_PERIOD_REVIEW over SAFETY_STOCK_ROP
        if "review every" in text or "review period" in text or "order-up-to" in text:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 5.0

        # If discount tiers are present, prefer QUANTITY_DISCOUNT over pure EOQ
        if scores[InventoryModelType.QUANTITY_DISCOUNT] >= 3.0 and ("discount" in text or "tier" in text or "break" in text):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 4.0

        best_model = max(scores, key=lambda m: scores[m])
        best_score = scores[best_model]

        if best_score < 1.5:
            return (
                InventoryModelType.UNKNOWN,
                0.2,
                "The question did not contain enough specific inventory keywords to decisively identify a model."
            )

        total_score = sum(scores.values())
        confidence = min(0.99, max(0.50, best_score / (total_score + 1e-6)))

        rationale = f"Selected {best_model.value} based on prominent matching terminology and query structure."
        return (best_model, confidence, rationale)

    # ========================================================================
    # STAGE 3: Extract Inputs
    # ========================================================================

    def _clean_number(self, val_str: str) -> Optional[float]:
        """Helper to safely parse cleaned float from regex capture."""
        if not val_str:
            return None
        cleaned = val_str.replace(",", "").replace("$", "").rstrip(".,;:!?% \t\n").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract a float matching any of the regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = self._clean_number(match.group(1))
                if val is not None:
                    return val
        return None

    def _extract_percentage(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract a percentage e.g. 95% -> 0.95."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_val = match.group(1).replace("%", "").rstrip(".,;:!? \t\n").strip()
                try:
                    val = float(raw_val.replace(",", ""))
                    return val / 100.0 if val > 1.0 else val
                except ValueError:
                    continue
        return None

    def _extract_ordering_cost(self, text: str) -> Optional[float]:
        """
        Extracts ordering/setup cost adhering to standard inventory terminology:
        - ordering cost
        - order cost
        - setup cost
        - setup expense
        - cost per order
        - cost of placing an order
        - ordering/setup cost
        And also handles inverted phrasings (e.g. '$50 per order', 'costs $50 to place').
        """
        patterns = [
            # 1. ordering/setup cost (with or without spaces/slashes)
            r"(?:ordering\s*/\s*setup\s+cost|ordering\s+and\s+setup\s+cost)\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 2. cost of placing an order
            r"cost\s+of\s+placing\s+(?:an?\s+)?order\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 3. cost per order
            r"cost\s+per\s+order\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 4. setup expense
            r"setup\s+expense\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 5. setup cost
            r"setup\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 6. order cost
            r"(?<!to\s)order\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # 7. ordering cost
            r"ordering\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            # Inverted phrasings:
            r"(?:each\s+)?order\s+costs?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:to\s+place)?",
            r"costs?\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(?:to\s+place(?:\s+an?\s+order)?|per\s+order|to\s+order|per\s+setup|to\s+set\s+up)",
            r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order|to\s+place\s+(?:an?\s+)?order|per\s+order|per\s+setup)",
            r"(?:order|ordering|setup)\s+(?:fee|charge)\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
        ]
        return self._extract_number(text, patterns)

    def _extract_demand_with_time_unit(self, text: str, default_work_days: int = 250) -> Dict[str, Any]:
        """
        Extracts demand rate and inspects associated time unit:
        - month / monthly / /month -> converts to annual by multiplying by 12
        - week / weekly / /week -> converts to annual by multiplying by 52 (or work_days / 5 if specified)
        - day / daily / /day -> converts to annual by multiplying by work_days (default 250, or 365 for calendar days)
        - quarter / quarterly / /quarter -> converts to annual by multiplying by 4
        - year / annually / /year -> 1.0 multiplier
        Returns:
            Dict containing:
            - annual_demand: float
            - stated_demand: float
            - stated_unit: str ('month', 'week', 'day', 'quarter', 'year')
            - conversion_note: Optional[str] (e.g. '1,000 units/month × 12 = 12,000 units/year')
            - multiplier: float
            - work_days: int
        """
        # Determine operating / working days
        if re.search(r"\b(calendar\s+days?|365\s*(?:days?|work\s*days?)|every\s+day\s+of\s+the\s+year|365\s*days?\s*per\s*year|seven\s+days\s+a\s+week|7\s*days\s*a\s*week)\b", text, re.IGNORECASE):
            work_days = 365
        else:
            m_wd = re.search(r"(\d+)\s*(?:working\s+days|operating\s+days|work\s+days|business\s+days|days\s+per\s+year|days\s+a\s+year)\b", text, re.IGNORECASE)
            work_days = int(m_wd.group(1)) if m_wd else default_work_days

        def clean_num(val_str: str) -> Optional[float]:
            try:
                return float(val_str.replace(",", "").strip())
            except ValueError:
                return None

        def norm_unit(u_str: str) -> str:
            u = u_str.lower().strip()
            if any(w in u for w in ["month", "mo"]):
                return "month"
            if any(w in u for w in ["week", "wk"]):
                return "week"
            if any(w in u for w in ["day", "daily"]):
                return "day"
            if any(w in u for w in ["quarter", "qtr"]):
                return "quarter"
            if any(w in u for w in ["year", "annum", "annual", "yr"]):
                return "year"
            return "year"

        def build_dict(val: float, unit: str, w_days: int) -> Dict[str, Any]:
            val_disp = f"{val:,.0f}" if val.is_integer() else f"{val:,.2f}"
            if unit == "month":
                annual = val * 12.0
                ann_disp = f"{annual:,.0f}" if annual.is_integer() else f"{annual:,.2f}"
                note = f"{val_disp} units/month × 12 = {ann_disp} units/year"
                multiplier = 12.0
            elif unit == "week":
                annual = val * 52.0
                ann_disp = f"{annual:,.0f}" if annual.is_integer() else f"{annual:,.2f}"
                note = f"{val_disp} units/week × 52 = {ann_disp} units/year"
                multiplier = 52.0
            elif unit == "day":
                annual = val * float(w_days)
                ann_disp = f"{annual:,.0f}" if annual.is_integer() else f"{annual:,.2f}"
                day_type = "calendar days" if w_days == 365 else "work days"
                note = f"{val_disp} units/day × {w_days} {day_type}/year = {ann_disp} units/year"
                multiplier = float(w_days)
            elif unit == "quarter":
                annual = val * 4.0
                ann_disp = f"{annual:,.0f}" if annual.is_integer() else f"{annual:,.2f}"
                note = f"{val_disp} units/quarter × 4 = {ann_disp} units/year"
                multiplier = 4.0
            else:  # "year"
                annual = val
                note = None
                multiplier = 1.0

            return {
                "annual_demand": annual,
                "stated_demand": val,
                "stated_unit": unit,
                "conversion_note": note,
                "multiplier": multiplier,
                "work_days": w_days,
            }

        # 1. Prefix time unit: e.g. "monthly demand is 1,000", "daily demand of 50"
        prefix_patterns = [
            r"\b(annual|yearly|monthly|weekly|daily|quarterly)\s+(?:demand|usage|sales|requirement|consumption)\s*(?:is|of|=|:|\baverages\b)?\s*([\d,]+(?:\.\d+)?)",
            r"\b(annual|yearly|monthly|weekly|daily|quarterly)\s+rate\s+of\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in prefix_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                unit = norm_unit(m.group(1))
                val = clean_num(m.group(2))
                if val is not None and val > 0:
                    return build_dict(val, unit, work_days)

        # 2. Postfix time unit directly attached to demand:
        postfix_patterns = [
            r"(?:demand|usage|sales|requirement|consumption)\s*(?:is|of|=|:|\baverages\b)?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items|cases|pieces|parts|boxes)?\s*(?:per\s+(?:year|annum|month|week|day|quarter)|\s*\/\s*(?:year|month|week|day|quarter|yr|mo|wk)|\b(?:a|each)\s+(?:year|month|week|day|quarter)\b|\b(?:annually|yearly|monthly|weekly|daily|quarterly)\b)",
            r"([\d,]+(?:\.\d+)?)\s*(?:units|items|cases|pieces|parts|boxes)\s*(?:per\s+(?:year|annum|month|week|day|quarter)|\s*\/\s*(?:year|month|week|day|quarter|yr|mo|wk)|\b(?:a|each)\s+(?:year|month|week|day|quarter)\b|\b(?:annually|yearly|monthly|weekly|daily|quarterly)\b)",
        ]
        for pat in postfix_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                matched_str = m.group(0)
                val = clean_num(m.group(1))
                unit = "year"
                if re.search(r"per\s+month|\/month|\/mo|\b(?:a|each)\s+month\b|\bmonthly\b", matched_str, re.IGNORECASE):
                    unit = "month"
                elif re.search(r"per\s+week|\/week|\/wk|\b(?:a|each)\s+week\b|\bweekly\b", matched_str, re.IGNORECASE):
                    unit = "week"
                elif re.search(r"per\s+day|\/day|\b(?:a|each)\s+day\b|\bdaily\b", matched_str, re.IGNORECASE):
                    unit = "day"
                elif re.search(r"per\s+quarter|\/quarter|\/qtr|\b(?:a|each)\s+quarter\b|\bquarterly\b", matched_str, re.IGNORECASE):
                    unit = "quarter"
                elif re.search(r"per\s+(?:year|annum)|\/(?:year|yr)|\b(?:a|each)\s+year\b|\b(?:annually|yearly)\b", matched_str, re.IGNORECASE):
                    unit = "year"
                if val is not None and val > 0:
                    return build_dict(val, unit, work_days)

        # 3. Standard / general demand patterns without explicit time unit (assumed annual in EOQ context)
        general_patterns = [
            r"annual\s+demand\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"demand\s*(?:is|of|=|:|\baverages\b)?\s*([\d,]+(?:\.\d+)?)",
            r"demand\s+of\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in general_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = clean_num(m.group(1))
                if val is not None and val > 0:
                    return build_dict(val, "year", work_days)

        return {
            "annual_demand": None,
            "stated_demand": None,
            "stated_unit": None,
            "conversion_note": None,
            "multiplier": 1.0,
            "work_days": work_days,
        }

    def _extract_annual_demand(self, text: str) -> Optional[float]:
        """Extracts annual / yearly demand rate, auto-converting non-annual units."""
        return self._extract_demand_with_time_unit(text)["annual_demand"]

    def _extract_holding_cost(
        self,
        text: str,
        work_days: int = 250,
        return_note: bool = False,
    ) -> Any:
        """
        Extracts holding / carrying cost, distinguishing:
        - fixed holding cost ($/unit/year)
        - percentage holding rate (% of purchase price)
        - time-unit conversions (per month, per week, per day, per quarter -> normalized to annual)
        Returns:
            If return_note=False: (fixed_holding_cost, percentage_holding_rate)
            If return_note=True: (fixed_holding_cost, percentage_holding_rate, conversion_note)
        """
        def clean_num(val_str: str) -> Optional[float]:
            try:
                return float(val_str.replace(",", "").strip())
            except ValueError:
                return None

        conv_note = None

        # 1. Check for percentage holding cost rate
        pct_patterns = [
            r"(?:holding|carrying)\s+(?:rate|cost|charge|expense)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*%",
            r"([\d,]+(?:\.\d+)?)\s*%\s*(?:annual\s+)?(?:holding|carrying)\s*(?:rate|cost|charge)?",
            r"([\d,]+(?:\.\d+)?)\s*%\s*of\s+(?:the\s+)?(?:unit\s+)?(?:price|cost)",
            r"holding\s+cost\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*%\s*(?:of\s+(?:unit\s+)?(?:price|cost))?",
        ]
        pct = None
        for pat in pct_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = clean_num(m.group(1))
                if val is not None:
                    raw_rate = val / 100.0 if val > 1.0 else val
                    # Check if percentage has time unit attached (e.g. per month)
                    post_context = text[m.start():m.end() + 25]
                    if re.search(r"%\s*(?:per\s+month|\/month|\/mo|\bmonthly\b)", post_context, re.IGNORECASE):
                        pct = raw_rate * 12.0
                        conv_note = f"{raw_rate*100:.1f}%/month × 12 = {pct*100:.1f}%/year"
                    elif re.search(r"%\s*(?:per\s+week|\/week|\/wk|\bweekly\b)", post_context, re.IGNORECASE):
                        pct = raw_rate * 52.0
                        conv_note = f"{raw_rate*100:.1f}%/week × 52 = {pct*100:.1f}%/year"
                    else:
                        pct = raw_rate
                    break

        # 2. Check for non-annual dollar holding cost (month, week, day, quarter)
        dollar = None
        dollar_time_patterns = [
            (r"(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit)?\s*(?:per\s+month|\/month|\/mo|\b(?:a|each)\s+month\b|\bmonthly\b)", "month"),
            (r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+month|per\s+unit\/month|\/unit\/month)", "month"),
            (r"(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit)?\s*(?:per\s+week|\/week|\/wk|\b(?:a|each)\s+week\b|\bweekly\b)", "week"),
            (r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+week|per\s+unit\/week|\/unit\/week)", "week"),
            (r"(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit)?\s*(?:per\s+day|\/day|\b(?:a|each)\s+day\b|\bdaily\b)", "day"),
            (r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+day|per\s+unit\/day|\/unit\/day)", "day"),
            (r"(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit)?\s*(?:per\s+quarter|\/quarter|\/qtr|\b(?:a|each)\s+quarter\b|\bquarterly\b)", "quarter"),
            (r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+quarter|per\s+unit\/quarter|\/unit\/quarter)", "quarter"),
        ]
        for pat, unit in dollar_time_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = clean_num(m.group(1))
                if val is not None and val > 0:
                    if unit == "month":
                        dollar = val * 12.0
                        conv_note = f"\\${val:,.2f}/unit/month × 12 = \\${dollar:,.2f}/unit/year"
                    elif unit == "week":
                        dollar = val * 52.0
                        conv_note = f"\\${val:,.2f}/unit/week × 52 = \\${dollar:,.2f}/unit/year"
                    elif unit == "day":
                        dollar = val * float(work_days)
                        conv_note = f"\\${val:,.2f}/unit/day × {work_days} = \\${dollar:,.2f}/unit/year"
                    elif unit == "quarter":
                        dollar = val * 4.0
                        conv_note = f"\\${val:,.2f}/unit/quarter × 4 = \\${dollar:,.2f}/unit/year"
                    break

        # 3. Standard annual dollar holding cost
        if dollar is None and pct is None:
            general_dollar_patterns = [
                r"(?:fixed\s+)?(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)(?!\s*%)",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+year|per\s+unit\s+annually|per\s+unit\/year)",
                r"costs?\s+\$?\s*([\d,]+(?:\.\d+)?)\s*to\s+(?:hold|carry)\s+(?:each|a|per)\s+unit",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:holding|carrying)\s+cost\s+per\s+unit",
                r"(?:holding|carrying)\s+expense\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)(?!\s*%)",
            ]
            dollar = self._extract_number(text, general_dollar_patterns)

        if return_note:
            return dollar, pct, conv_note
        return dollar, pct

    def _extract_unit_price(self, text: str) -> Optional[float]:
        """Extracts unit purchase price / product cost, avoiding holding or ordering cost matches."""
        patterns = [
            r"(?:unit\s+(?:purchase\s+)?(?:price|cost)|purchase\s+price|unit\s+price|product\s+cost|item\s+cost|wholesale\s+price)\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            r"buy(?:ing)?\s+(?:them\s+)?(?:for|at)\s+\$?\s*([\d,]+(?:\.\d+)?)",
            r"costs?\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|each|a\s+unit|to\s+buy|to\s+purchase)(?!\s*(?:per\s+year|annually|\/year|to\s+hold|holding))",
            r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|each|a\s+unit)(?!\s*(?:per\s+year|annually|\/year|to\s+hold|holding|order|setup))",
        ]
        return self._extract_number(text, patterns)

    def _extract_lead_time_parameters(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """
        Distinguishes:
        - average lead time
        - lead-time standard deviation
        Returns: (avg_lead_time, std_lead_time, period_name)
        """
        period_name = "day"
        if re.search(r"\bweeks?\b", text, re.IGNORECASE):
            period_name = "week"
        elif re.search(r"\bmonths?\b", text, re.IGNORECASE):
            period_name = "month"

        # Lead time std dev
        std_patterns = [
            r"(?:lead[- ]time\s+standard\s+deviation|standard\s+deviation\s+of\s+lead[- ]time|lead[- ]time\s+std\s+dev|std\s+dev\s+of\s+lead[- ]time|lead[- ]time\s+sigma)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"lead[- ]time\s+(?:varies\s+with\s+(?:a\s+)?|has\s+(?:a\s+)?)(?:standard\s+deviation|std\s+dev)\s*(?:of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:with\s+a\s+)?(?:std\s+dev|standard\s+deviation)\s+in\s+lead[- ]time\s+(?:of\s+)?([\d,]+(?:\.\d+)?)",
        ]
        std_lt = self._extract_number(text, std_patterns)

        # Average lead time
        avg_patterns = [
            r"(?:average|mean|expected)\s+lead[- ]time\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:supplier|replenishment|delivery)?\s*lead[- ]time\s*(?:averages|is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?|months?)?",
            r"([\d,]+(?:\.\d+)?)\s*[- ](?:day|week|month)\s+lead[- ]time",
            r"lead[- ]time\s+(?:of|=|:)\s*([\d,]+(?:\.\d+)?)",
        ]
        avg_lt = self._extract_number(text, avg_patterns)

        return avg_lt, std_lt, period_name

    def _extract_demand_parameters(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """
        Distinguishes:
        - average demand
        - demand standard deviation
        Returns: (avg_demand, std_demand, period_name)
        """
        period_name = "day"
        if re.search(r"\bweeks?\b", text, re.IGNORECASE):
            period_name = "week"
        elif re.search(r"\bmonths?\b", text, re.IGNORECASE):
            period_name = "month"

        # Demand std dev
        std_patterns = [
            r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s+dev|std\s+dev\s+of\s+demand|demand\s+sigma)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:with\s+(?:a\s+)?)?(?:standard\s+deviation|std\s+dev|sigma)\s+(?:of\s+demand\s+)?(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"demand\s+(?:varies\s+with\s+(?:a\s+)?|has\s+(?:a\s+)?)(?:standard\s+deviation|std\s+dev)\s*(?:of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"std\s+dev\s*[:=]?\s*([\d,]+(?:\.\d+)?)",
        ]
        std_d = self._extract_number(text, std_patterns)

        # Average demand
        avg_patterns = [
            r"(?:average|mean|expected)\s+(?:daily|weekly|monthly)?\s*demand\s*(?:averages|is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:daily|weekly|monthly)\s+demand\s*(?:averages|is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"demand\s+(?:averages|is|of|=|:)\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?",
            r"normally\s+distributed\s+with\s+mean\s*(?:of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"mean\s*(?:of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units)?",
            r"([\d,]+(?:\.\d+)?)\s*units\s*per\s+(?:day|week|month)",
        ]
        avg_d = self._extract_number(text, avg_patterns)

        return avg_d, std_d, period_name

    def _extract_service_level(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts service level (CSL) as fraction and/or z-score.
        Returns: (service_level, z_score)
        """
        csl = self._extract_percentage(text, [
            r"(?:cycle\s+service\s+level|service\s+level|csl|target\s+service\s+level|fill\s+rate)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*%",
            r"([\d,]+(?:\.\d+)?)\s*%\s*(?:cycle\s+service\s+level|service\s+level|csl)",
        ])

        z = self._extract_number(text, [
            r"z(?:[- ]score)?\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"z\s*=\s*([\d,]+(?:\.\d+)?)",
        ])

        return csl, z

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts required and optional parameters from the natural language question.
        Returns: (extracted_dict, missing_required_fields)
        """
        text = question
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        if model_type == InventoryModelType.EOQ:
            d_info = self._extract_demand_with_time_unit(text)
            d = d_info["annual_demand"]
            s = self._extract_ordering_cost(text)
            work_days = d_info["work_days"]
            f_h, pct_h, h_note = self._extract_holding_cost(text, work_days=work_days, return_note=True)
            c = self._extract_unit_price(text)
            lt_mean, _, _ = self._extract_lead_time_parameters(text)

            # Determine holding cost: fixed dollar amount or percentage of unit purchase price
            h = None
            if f_h is not None:
                h = f_h
            elif pct_h is not None and c is not None and c > 0:
                h = pct_h * c
            elif pct_h is not None:
                pass

            if d is not None:
                inputs["annual_demand"] = d
                if d_info.get("conversion_note"):
                    inputs["demand_conversion"] = d_info["conversion_note"]
            else:
                missing.append("annual_demand")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost")

            if h is not None:
                inputs["holding_cost"] = h
                if h_note:
                    inputs["holding_conversion"] = h_note
            elif pct_h is not None and c is None:
                missing.append("unit_cost (required to calculate holding cost from percentage rate)")
            else:
                missing.append("holding_cost")

            inputs["unit_cost"] = c if c is not None else 0.0
            inputs["lead_time_days"] = lt_mean if lt_mean is not None else 0.0
            inputs["work_days_per_year"] = work_days

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            d_info = self._extract_demand_with_time_unit(text)
            d = d_info["annual_demand"]
            s = self._extract_ordering_cost(text)
            work_days = d_info["work_days"]
            f_h, pct_h, h_note = self._extract_holding_cost(text, work_days=work_days, return_note=True)
            unit_price = self._extract_unit_price(text)

            # Extract discount tiers
            tiers: List[DiscountTier] = []

            # Pattern A: e.g. "under 500 units @ $10", "1 to 499 units @ $10.00", "1000+ units @ $9.00"
            tier_candidates = re.findall(
                r"(?:(?:under|<|less\s+than|up\s+to|0\s*(?:-|to))\s*([\d,]+)\s*(?:units)?\s*(?:@|cost|costs|for|at|:|=)?\s*\$?\s*(\d+(?:\.\d+)?))|"
                r"(?:([\d,]+)\s*(?:-|–|to)\s*([\d,]+)\s*(?:units)?\s*(?:@|cost|costs|for|at|:|=)?\s*\$?\s*(\d+(?:\.\d+)?))|"
                r"(?:([\d,]+)\s*(?:\+|or\s+more|and\s+above|\s*and\s*over)\s*(?:units)?\s*(?:@|cost|costs|for|at|:|=)?\s*\$?\s*(\d+(?:\.\d+)?))",
                text,
                re.IGNORECASE
            )

            if tier_candidates:
                for group in tier_candidates:
                    if group[0] and group[1]:
                        max_q = float(group[0].replace(",", "")) - 1.0
                        price = float(group[1].replace(",", "").rstrip(".,;:!?"))
                        tiers.append(DiscountTier(min_qty=1.0, max_qty=max_q, unit_price=price, tier_name=f"1 to {int(max_q)}"))
                    elif group[2] and group[3] and group[4]:
                        min_q = float(group[2].replace(",", ""))
                        max_q = float(group[3].replace(",", ""))
                        price = float(group[4].replace(",", "").rstrip(".,;:!?"))
                        tiers.append(DiscountTier(min_qty=min_q, max_qty=max_q, unit_price=price, tier_name=f"{int(min_q)} to {int(max_q)}"))
                    elif group[5] and group[6]:
                        min_q = float(group[5].replace(",", ""))
                        price = float(group[6].replace(",", "").rstrip(".,;:!?"))
                        tiers.append(DiscountTier(min_qty=min_q, max_qty=None, unit_price=price, tier_name=f"{int(min_q)}+"))

            # Pattern B: table-like formats: e.g. "1-499: $10, 500-999: $9.50, 1000+: $9.00"
            if not tiers:
                table_matches = re.findall(
                    r"([\d,]+)\s*(?:-|–|to)\s*([\d,]+|\+)?\s*(?:units)?\s*[:@=]\s*\$?\s*(\d+(?:\.\d+)?)",
                    text,
                    re.IGNORECASE
                )
                for m_min, m_max, m_price in table_matches:
                    min_q = float(m_min.replace(",", ""))
                    max_q = float(m_max.replace(",", "")) if m_max and m_max != "+" else None
                    price = float(m_price.replace(",", "").rstrip(".,;:!?"))
                    tier_name = f"{int(min_q)} to {int(max_q)}" if max_q else f"{int(min_q)}+"
                    tiers.append(DiscountTier(min_qty=min_q, max_qty=max_q, unit_price=price, tier_name=tier_name))

            # Pattern C: Price breaks with quantities: e.g. "$10 for 1-499, $9.50 for 500-999, $9 for 1000+"
            if not tiers:
                inv_matches = re.findall(
                    r"\$?\s*(\d+(?:\.\d+)?)\s*(?:for|at)?\s*(?:orders\s+of\s+)?([\d,]+)\s*(?:-|–|to)\s*([\d,]+|\+)?\s*(?:units)?",
                    text,
                    re.IGNORECASE
                )
                for m_price, m_min, m_max in inv_matches:
                    price = float(m_price.replace(",", "").rstrip(".,;:!?"))
                    min_q = float(m_min.replace(",", ""))
                    max_q = float(m_max.replace(",", "")) if m_max and m_max != "+" else None
                    tier_name = f"{int(min_q)} to {int(max_q)}" if max_q else f"{int(min_q)}+"
                    tiers.append(DiscountTier(min_qty=min_q, max_qty=max_q, unit_price=price, tier_name=tier_name))

            # Sort tiers by min_qty
            tiers.sort(key=lambda t: t.min_qty)

            if d is not None:
                inputs["annual_demand"] = d
                if d_info.get("conversion_note"):
                    inputs["demand_conversion"] = d_info["conversion_note"]
            else:
                missing.append("annual_demand")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost")

            if pct_h is not None:
                inputs["holding_rate"] = pct_h
                if h_note:
                    inputs["holding_conversion"] = h_note
            elif f_h is not None:
                inputs["fixed_holding_cost"] = f_h
                if h_note:
                    inputs["holding_conversion"] = h_note
            else:
                missing.append("holding_rate or fixed_holding_cost")

            if tiers:
                inputs["tiers"] = tiers
            else:
                missing.append("discount_tiers")

            if unit_price is not None:
                inputs["unit_purchase_price"] = unit_price

            inputs["work_days_per_year"] = work_days

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            d_info = self._extract_demand_with_time_unit(text)
            avg_d, std_d, period_d = self._extract_demand_parameters(text)
            avg_lt, std_lt, period_lt = self._extract_lead_time_parameters(text)
            csl, z_val = self._extract_service_level(text)

            period = period_d if period_d != "day" else period_lt
            demand_conv = d_info.get("conversion_note")
            work_days = d_info.get("work_days", 250)

            if avg_d is None and d_info.get("annual_demand") is not None:
                ann_d = d_info["annual_demand"]
                if period == "day":
                    avg_d = ann_d / float(work_days)
                elif period == "week":
                    avg_d = ann_d / 52.0
                elif period == "month":
                    avg_d = ann_d / 12.0
                else:
                    avg_d = ann_d

            if d_info.get("annual_demand") is not None:
                inputs["annual_demand"] = d_info["annual_demand"]
            if demand_conv:
                inputs["demand_conversion"] = demand_conv

            # Check whether problem context implies stochastic variability or deterministic constant parameters
            has_variability = (
                std_d is not None
                or std_lt is not None
                or csl is not None
                or z_val is not None
                or bool(re.search(r"\b(safety\s+stock|service\s+level|csl|z[- ]score|uncertain|variable|stochastic)\b", text, re.IGNORECASE))
            )

            if avg_d is not None:
                inputs["avg_demand_per_period"] = avg_d
            else:
                missing.append("avg_demand_per_period")

            if avg_lt is not None:
                inputs["lead_time_periods"] = avg_lt
            else:
                missing.append("lead_time_periods")

            # Demand standard deviation: if demand is constant and no variability is stated, default to 0.0
            if std_d is not None:
                inputs["std_demand_per_period"] = std_d
            elif has_variability:
                missing.append("std_demand_per_period")
            else:
                inputs["std_demand_per_period"] = 0.0

            # Lead time standard deviation
            inputs["std_lead_time_periods"] = std_lt if std_lt is not None else 0.0

            # Service level or z-score
            if csl is not None:
                inputs["service_level"] = csl
            elif z_val is not None:
                inputs["service_level"] = NormalDist(0.0, 1.0).cdf(z_val)
                inputs["z_score"] = z_val
            elif has_variability:
                inputs["service_level"] = 0.95
            else:
                inputs["service_level"] = 0.95

            inputs["period_name"] = period

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            d_info = self._extract_demand_with_time_unit(text)
            # Review period R
            r = self._extract_number(text, [
                r"review\s+(?:period|interval)?\s*(?:is|every|of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)?",
                r"reviewed?\s+every\s+(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)?",
                r"every\s+(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
                r"order\s+every\s+(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
            ])
            # Lead time L
            lt_mean, _, period_lt = self._extract_lead_time_parameters(text)
            # Demand rate and std dev
            avg_d, std_d, period_d = self._extract_demand_parameters(text)
            # Service level
            csl, z_val = self._extract_service_level(text)

            period = period_d if period_d != "day" else period_lt
            demand_conv = d_info.get("conversion_note")
            work_days = d_info.get("work_days", 250)

            if avg_d is None and d_info.get("annual_demand") is not None:
                ann_d = d_info["annual_demand"]
                if period == "day":
                    avg_d = ann_d / float(work_days)
                elif period == "week":
                    avg_d = ann_d / 52.0
                elif period == "month":
                    avg_d = ann_d / 12.0
                else:
                    avg_d = ann_d

            if d_info.get("annual_demand") is not None:
                inputs["annual_demand"] = d_info["annual_demand"]
            if demand_conv:
                inputs["demand_conversion"] = demand_conv

            # Safety stock directly specified (e.g. "safety stock of 50 units")
            ss_direct = self._extract_number(text, [
                r"(?:safety\s+stock|buffer\s+stock)\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)",
                r"keep\s+(?:a\s+)?safety\s+stock\s+of\s*(\d+(?:\.\d+)?)",
            ])

            # Inventory position directly specified
            inv_pos_direct = self._extract_number(text, [
                r"(?:current\s+)?inventory\s+position\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)",
            ])

            # Current on-hand
            on_hand = self._extract_number(text, [
                r"on[- ]hand(?:\s+inventory)?\s*(?:we\s+have|is|=|:|\s+of)?\s*([\d,]+(?:\.\d+)?)",
                r"(?:currently\s+)?have\s+([\d,]+(?:\.\d+)?)\s*(?:units)?\s*on[- ]hand",
                r"([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:currently\s+)?on[- ]hand",
                r"(?:current\s+inventory|stock\s+on\s+hand)\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"in\s+stock\s*(?:we\s+have|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+)?in\s+stock",
            ])

            # Current on-order
            on_order = self._extract_number(text, [
                r"on[- ]order(?:\s+inventory)?\s*(?:we\s+have|is|=|:|\s+of)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:currently\s+)?on[- ]order",
                r"(?:open\s+orders?|in[- ]transit)\s*(?:is|=|:|\s+of)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+)?(?:open\s+orders?|in[- ]transit)",
            ])

            # Backorders
            backorders = self._extract_number(text, [
                r"backorders?\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+)?(?:backorders?|on\s+backorder|backordered)",
                r"(?:backlog|backlogged)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+)?(?:backlog|backlogged)",
            ])

            if r is not None:
                inputs["review_period"] = r
            else:
                missing.append("review_period")

            if lt_mean is not None:
                inputs["lead_time"] = lt_mean
            else:
                missing.append("lead_time")

            if avg_d is not None:
                inputs["avg_demand_per_period"] = avg_d
            else:
                missing.append("avg_demand_per_period")

            if std_d is not None:
                inputs["std_demand_per_period"] = std_d
            elif ss_direct is not None:
                inputs["std_demand_per_period"] = 0.0
            else:
                inputs["std_demand_per_period"] = 0.0

            if ss_direct is not None:
                inputs["safety_stock"] = ss_direct

            if csl is not None:
                inputs["service_level"] = csl
            elif z_val is not None:
                inputs["service_level"] = NormalDist(0.0, 1.0).cdf(z_val)
            else:
                inputs["service_level"] = 0.95

            oh = on_hand if on_hand is not None else 0.0
            oo = on_order if on_order is not None else 0.0
            bo = backorders if backorders is not None else 0.0

            inputs["current_on_hand"] = oh
            inputs["current_on_order"] = oo
            inputs["backorders"] = bo

            if inv_pos_direct is not None:
                inputs["inventory_position"] = inv_pos_direct
            else:
                inputs["inventory_position"] = oh + oo - bo

            inputs["period_name"] = period

        elif model_type == InventoryModelType.NEWSVENDOR:
            # 1. Underage cost / Shortage cost (Cu)
            cu = self._extract_number(text, [
                r"(?:cost\s+of\s+underage|underage\s+cost|shortage\s+cost|cost\s+of\s+shortage|unmet\s+demand\s+cost)\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"\$?\s*(\d+(?:\.\d+)?)\s*(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|loss\s+per\s+unmet\s+unit)",
            ])

            # 2. Overage cost / Excess cost (Co)
            co = self._extract_number(text, [
                r"(?:cost\s+of\s+overage|overage\s+cost|excess\s+cost|cost\s+of\s+excess|leftover\s+cost|disposal\s+cost)\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"\$?\s*(\d+(?:\.\d+)?)\s*(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|loss\s+per\s+leftover\s+unit)",
            ])

            # 3. Selling price (P)
            p = self._extract_number(text, [
                r"sell(?:ing)?\s+(?:price|for)?\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"(?:retail\s+price|sale\s+price|sales\s+price|revenue\s+per\s+unit)\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"sold\s+(?:at|for)\s*\$?\s*(\d+(?:\.\d+)?)",
                r"for\s+\$?\s*(\d+(?:\.\d+)?)\s*each",
                r"price\s*(?:is|of|=|:)\s*\$?\s*(\d+(?:\.\d+)?)",
            ])

            # 4. Purchase cost / unit cost (C)
            c = self._extract_number(text, [
                r"(?:purchase\s+cost|purchase\s+price|unit\s+cost|acquisition\s+cost|wholesale\s+cost|cost\s+to\s+produce|cost\s+to\s+buy)\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"buy\s+(?:them\s+)?(?:for|at)\s*\$?\s*(\d+(?:\.\d+)?)",
                r"costs?\s+\$?\s*(\d+(?:\.\d+)?)\s*(?:to\s+make|to\s+produce|to\s+buy|per\s+unit|each)",
                r"(?<!holding\s)(?<!order\s)(?<!ordering\s)cost\s*(?:is|of|=|:)\s*\$?\s*(\d+(?:\.\d+)?)",
            ])

            # 5. Salvage value (V)
            v = self._extract_number(text, [
                r"(?:salvage|markdown|liquidation|scrap|clearance|residual)\s+(?:value|price)?\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
                r"liquidated?\s+for\s+\$?\s*(\d+(?:\.\d+)?)",
                r"sold\s+at\s+markdown\s+for\s+\$?\s*(\d+(?:\.\d+)?)",
                r"scrap\s+for\s+\$?\s*(\d+(?:\.\d+)?)",
            ])

            # 6. Stockout penalty (g)
            g = self._extract_number(text, [
                r"(?:stockout\s+penalty|penalty\s+cost|goodwill\s+loss|loss\s+of\s+goodwill|penalty)\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:\.\d+)?)",
            ])

            # 7. Demand distribution & parameters
            avg_d, std_d, _ = self._extract_demand_parameters(text)

            dist_name = "normal"
            if re.search(r"uniform", text, re.IGNORECASE):
                dist_name = "uniform"

            inputs["demand_distribution"] = dist_name

            # Examine the entire question before declaring information missing:
            # If Cu and Co are directly provided, selling_price and unit_cost are not missing!
            if cu is not None and co is not None:
                inputs["underage_cost"] = cu
                inputs["overage_cost"] = co
                inputs["unit_cost"] = c if c is not None else co
                inputs["selling_price"] = p if p is not None else (cu + inputs["unit_cost"])
            else:
                if p is not None:
                    inputs["selling_price"] = p
                else:
                    missing.append("selling_price (or underage_cost)")

                if c is not None:
                    inputs["unit_cost"] = c
                else:
                    missing.append("unit_cost (or overage_cost)")

            inputs["salvage_value"] = v if v is not None else 0.0
            inputs["stockout_penalty"] = g if g is not None else 0.0

            if avg_d is not None:
                inputs["mean_demand"] = avg_d
            else:
                missing.append("mean_demand")

            if std_d is not None:
                inputs["std_demand"] = std_d
            else:
                missing.append("std_demand")

        return inputs, missing

    # ========================================================================
    # STAGE 4: Select Python Tool & Prepare Typed Inputs
    # ========================================================================

    def select_and_calculate_tool(self, model_type: InventoryModelType, inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Executes the corresponding pure Python tool in inventory_tools.py.
        Returns: (tool_name, calculation_result)
        """
        if model_type == InventoryModelType.EOQ:
            tool_name = "inventory_tools.calculate_eoq"
            eoq_in = EOQInputs(
                annual_demand=inputs["annual_demand"],
                order_cost=inputs["order_cost"],
                holding_cost=inputs["holding_cost"],
                unit_cost=inputs.get("unit_cost", 0.0),
                lead_time_days=inputs.get("lead_time_days", 0.0),
                work_days_per_year=inputs.get("work_days_per_year", 250),
                demand_conversion_note=inputs.get("demand_conversion"),
                holding_conversion_note=inputs.get("holding_conversion"),
            )
            result = calculate_eoq(eoq_in)
            return tool_name, result

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            tool_name = "inventory_tools.calculate_quantity_discount"
            qd_in = QuantityDiscountInputs(
                annual_demand=inputs["annual_demand"],
                order_cost=inputs["order_cost"],
                tiers=inputs["tiers"],
                holding_rate=inputs.get("holding_rate"),
                fixed_holding_cost=inputs.get("fixed_holding_cost"),
                work_days_per_year=inputs.get("work_days_per_year", 250),
                demand_conversion_note=inputs.get("demand_conversion"),
                holding_conversion_note=inputs.get("holding_conversion"),
            )
            result = calculate_quantity_discount(qd_in)
            return tool_name, result

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            tool_name = "inventory_tools.calculate_rop"
            ss_in = SafetyStockROPInputs(
                avg_demand_per_period=inputs["avg_demand_per_period"],
                std_demand_per_period=inputs["std_demand_per_period"],
                lead_time_periods=inputs["lead_time_periods"],
                std_lead_time_periods=inputs.get("std_lead_time_periods", 0.0),
                service_level=inputs.get("service_level", 0.95),
                period_name=inputs.get("period_name", "day"),
                demand_conversion_note=inputs.get("demand_conversion"),
            )
            result = calculate_rop(ss_in)
            return tool_name, result

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            tool_name = "inventory_tools.calculate_periodic_review"
            fp_in = FixedPeriodReviewInputs(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                avg_demand_per_period=inputs["avg_demand_per_period"],
                std_demand_per_period=inputs.get("std_demand_per_period", 0.0),
                service_level=inputs.get("service_level", 0.95),
                current_on_hand=inputs.get("current_on_hand", 0.0),
                current_on_order=inputs.get("current_on_order", 0.0),
                period_name=inputs.get("period_name", "day"),
                safety_stock=inputs.get("safety_stock"),
                inventory_position=inputs.get("inventory_position"),
                backorders=inputs.get("backorders", 0.0),
                demand_conversion_note=inputs.get("demand_conversion"),
            )
            result = calculate_periodic_review(fp_in)
            return tool_name, result

        elif model_type == InventoryModelType.NEWSVENDOR:
            tool_name = "inventory_tools.calculate_newsvendor"
            nv_in = NewsvendorInputs(
                unit_cost=inputs.get("unit_cost", 0.0),
                selling_price=inputs.get("selling_price", 0.0),
                salvage_value=inputs.get("salvage_value", 0.0),
                stockout_penalty=inputs.get("stockout_penalty", 0.0),
                mean_demand=inputs["mean_demand"],
                std_demand=inputs["std_demand"],
                underage_cost=inputs.get("underage_cost"),
                overage_cost=inputs.get("overage_cost"),
                critical_ratio=inputs.get("critical_ratio"),
            )
            result = calculate_newsvendor(nv_in)
            return tool_name, result

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # ========================================================================
    # STAGE 5 & 6: Verify Calculation Results
    # ========================================================================

    def verify_result(self, model_type: InventoryModelType, result: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Performs mathematical verification and invariant checks on the numerical result.
        Returns: (is_verified, verification_details)
        """
        if not hasattr(result, "verification"):
            return True, {"status": "No explicit verification payload attached."}

        details = dict(result.verification)
        is_verified = details.get("is_valid", False)
        return is_verified, details

    # ========================================================================
    # STAGE 7: Explain and Generate Actionable Business Insights
    # ========================================================================

    def explain_result(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
        is_verified: bool
    ) -> AgentExplanation:
        """
        Translates raw numerical calculations into structured executive explanations,
        mathematical formula breakdowns, and actionable operational recommendations.
        """
        if model_type == InventoryModelType.EOQ:
            r: EOQResult = result
            conv_prefix = ""
            if inputs.get("demand_conversion"):
                conv_prefix += f"\n- **Demand Annualization:** `{inputs['demand_conversion']}`"
            if inputs.get("holding_conversion"):
                conv_prefix += f"\n- **Holding Cost Annualization:** `{inputs['holding_conversion']}`"

            exec_sum = (
                f"**Optimal Order Quantity (EOQ):** `{r.eoq:,.1f} units` per order."
                f"{conv_prefix}\n"
                f"- **Order Frequency:** `{r.orders_per_year:.2f} orders/year` (~every `{r.order_interval_days:.1f} working days`).\n"
                f"- **Total Annual Inventory Cost:** `${r.total_annual_inventory_cost:,.2f}` "
                f"(${r.annual_ordering_cost:,.2f} ordering + ${r.annual_holding_cost:,.2f} holding)."
            )
            if r.reorder_point > 0:
                exec_sum += f"\n- **Reorder Point (ROP):** Place a replenishment order when inventory reaches `{r.reorder_point:,.1f} units`."

            formula_trace = []
            f_idx = 1
            if inputs.get("demand_conversion"):
                formula_trace.append(f"{f_idx}. **Demand Annualization:** {inputs['demand_conversion']}.")
                f_idx += 1
            if inputs.get("holding_conversion"):
                formula_trace.append(f"{f_idx}. **Holding Cost Annualization:** {inputs['holding_conversion']}.")
                f_idx += 1

            formula_trace.extend([
                f"{f_idx}. **EOQ Formula:** $Q^* = \\sqrt{{\\frac{{2DS}}{{H}}}} = \\sqrt{{\\frac{{2 \\times {inputs['annual_demand']:,.0f} \\times {inputs['order_cost']:,.2f}}}{{{inputs['holding_cost']:,.2f}}}}} = {r.eoq:,.2f}$ units.",
                f"{f_idx + 1}. **Orders per Year:** $N = \\frac{{D}}{{Q^*}} = \\frac{{{inputs['annual_demand']:,.0f}}}{{{r.eoq:,.2f}}} = {r.orders_per_year:.2f}$.",
                f"{f_idx + 2}. **Annual Ordering Cost:** $\\frac{{D}}{{Q^*}} S = {r.orders_per_year:.2f} \\times$ \\${inputs['order_cost']:,.2f} = \\${r.annual_ordering_cost:,.2f}.",
                f"{f_idx + 3}. **Annual Holding Cost:** $\\frac{{Q^*}}{{2}} H = \\frac{{{r.eoq:,.2f}}}{{2}} \\times$ \\${inputs['holding_cost']:,.2f} = \\${r.annual_holding_cost:,.2f}.",
                f"{f_idx + 4}. **Total Annual Inventory Cost:** $\\text{{AOC}} + \\text{{AHC}} =$ \\${r.total_annual_inventory_cost:,.2f}.",
            ])

            insights = [
                f"**Trade-off Equality:** At EOQ, annual ordering cost exactly equals annual holding cost (${r.annual_ordering_cost:,.2f} vs ${r.annual_holding_cost:,.2f}).",
                f"**Sensitivity / Robustness:** If order size deviates by +10% ({r.eoq*1.1:,.1f} units), total inventory cost increases by only ~{r.verification['sensitivity']['penalty_plus_10pct']*100:.2f}%.",
                f"**Purchase Cost Neutrality:** Total annual purchase spend is ${r.annual_purchase_cost:,.2f}, bringing total procurement cost to ${r.total_annual_cost:,.2f}.",
            ]

            rec_action = (
                f"Authorize purchase orders for batch sizes of **{round(r.eoq):,d} units**. "
                f"Schedule replenishment roughly every **{round(r.order_interval_days):d} business days**."
            )

            ver_sum = (
                f"Passed: Mathematical first-order condition AOC == AHC confirmed (diff: ${r.verification['cost_difference']:.4f}). "
                f"Strict global convex minimum verified (d²TC/dQ² > 0)."
            )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            opt_tier = r.tier_evaluations[r.optimal_tier_index].tier
            conv_prefix = ""
            if inputs.get("demand_conversion"):
                conv_prefix += f"\n- **Demand Annualization:** `{inputs['demand_conversion']}`"
            if inputs.get("holding_conversion"):
                conv_prefix += f"\n- **Holding Cost Annualization:** `{inputs['holding_conversion']}`"

            exec_sum = (
                f"**Optimal Order Quantity:** `{r.optimal_order_qty:,.1f} units` (qualifying for Tier {r.optimal_tier_index + 1}: "
                f"`{opt_tier.tier_name}` at `${r.optimal_unit_price:,.2f}/unit`)."
                f"{conv_prefix}\n"
                f"- **Total Annual Cost:** `${r.optimal_total_annual_cost:,.2f}` (Purchase: `${r.optimal_annual_purchase_cost:,.2f}`, "
                f"Order: `${r.optimal_annual_ordering_cost:,.2f}`, Holding: `${r.optimal_annual_holding_cost:,.2f}`).\n"
                f"- **Net Annual Savings vs Base Pricing:** `${r.annual_savings_vs_base:,.2f}` (`{r.percentage_savings_vs_base:.2f}%` total cost reduction)."
            )

            formula_trace = []
            if inputs.get("demand_conversion"):
                formula_trace.append(f"**Demand Annualization:** {inputs['demand_conversion']}.")
            if inputs.get("holding_conversion"):
                formula_trace.append(f"**Holding Cost Annualization:** {inputs['holding_conversion']}.")
            for idx, ev in enumerate(r.tier_evaluations):
                status_str = "Feasible EOQ" if ev.is_raw_eoq_feasible else "Adjusted to Tier Price Break"
                formula_trace.append(
                    f"Tier {idx+1} ({ev.tier.tier_name} @ \\${ev.tier.unit_price:,.2f}): Raw EOQ = {ev.raw_eoq:,.1f} units. "
                    f"Evaluated Q = {ev.feasible_order_qty:,.1f} ({status_str}) -> Total Cost = \\${ev.total_annual_cost:,.2f}."
                )

            insights = [
                f"**Volume Discount Exploitation:** Increasing batch size from base EOQ to the price break ({r.optimal_order_qty:,.0f} units) saves ${r.annual_savings_vs_base:,.2f} annually.",
                f"**Holding Cost Trade-off:** While annual holding cost increases to ${r.optimal_annual_holding_cost:,.2f}, the unit purchase savings of ${r.base_tier_cost - r.optimal_annual_purchase_cost - r.optimal_annual_ordering_cost - r.optimal_annual_holding_cost:,.2f} overwhelmingly compensate.",
                f"**Working Capital Impact:** Average cycle inventory will be {r.optimal_order_qty/2.0:,.1f} units, tying up ${(r.optimal_order_qty/2.0)*r.optimal_unit_price:,.2f} in working capital.",
            ]

            rec_action = (
                f"Execute orders at the tier threshold of **{round(r.optimal_order_qty):,d} units** "
                f"to lock in the ${r.optimal_unit_price:,.2f} unit price and capture ${r.annual_savings_vs_base:,.2f} in annual savings."
            )

            ver_sum = (
                f"Passed: All {r.verification['total_tiers_evaluated']} tiers evaluated. "
                f"Tier {r.optimal_tier_index + 1} confirmed globally minimal. Order quantity satisfies tier boundary constraints."
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            r: SafetyStockROPResult = result
            conv_str = f"\n- **Demand Normalization:** `{inputs['demand_conversion']}`" if inputs.get("demand_conversion") else ""
            exec_sum = (
                f"**Recommended Safety Stock (SS):** `{r.safety_stock:,.1f} units` (buffer stock)."
                f"{conv_str}\n"
                f"- **Reorder Point (ROP):** `{r.reorder_point:,.1f} units`.\n"
                f"- **Cycle Service Level:** `{inputs.get('service_level', 0.95)*100:.1f}%` (Stockout risk per cycle: `{r.stockout_risk_per_cycle*100:.1f}%`).\n"
                f"- **Expected Units Short per Cycle:** `{r.expected_short_per_cycle:.3f} units`."
            )

            formula_trace = [
                f"1. **Lead Time Demand Mean:** $\\mu_{{DLT}} = \\bar{{d}} \\times \\bar{{L}} = {inputs['avg_demand_per_period']:.1f} \\times {inputs['lead_time_periods']:.1f} = {r.lead_time_demand_mean:,.1f}$ units.",
                f"2. **Lead Time Demand Std Dev:** $\\sigma_{{DLT}} = \\sqrt{{\\bar{{L}} \\sigma_d^2 + \\bar{{d}}^2 \\sigma_L^2}} = {r.lead_time_demand_std:,.2f}$ units.",
                f"3. **Target Z-Score:** $z = \\Phi^{{-1}}({inputs.get('service_level', 0.95):.2f}) = {r.z_score:.4f}$.",
                f"4. **Safety Stock:** $SS = z \\times \\sigma_{{DLT}} = {r.z_score:.4f} \\times {r.lead_time_demand_std:,.2f} = {r.safety_stock:,.1f}$ units.",
                f"5. **Reorder Point:** $ROP = \\mu_{{DLT}} + SS = {r.lead_time_demand_mean:,.1f} + {r.safety_stock:,.1f} = {r.reorder_point:,.1f}$ units.",
            ]

            insights = [
                f"**Lead Time Vulnerability:** With average lead time of {inputs['lead_time_periods']} {r.period_name}s, expected consumption during replenishment is {r.lead_time_demand_mean:,.1f} units.",
                f"**Buffer Size:** A safety stock of {r.safety_stock:,.1f} units protects against demand surges up to the {inputs.get('service_level', 0.95)*100:.1f}th percentile.",
                f"**Stockout Impact:** Under this policy, on average only {r.expected_short_per_cycle:.3f} units are unfilled per replenishment cycle.",
            ]

            rec_action = (
                f"Configure ERP/WMS inventory trigger: place a replenishment order immediately when on-hand + on-order "
                f"inventory dips to or below **{round(r.reorder_point):,d} units**."
            )

            ver_sum = (
                f"Passed: Target service level {inputs.get('service_level', 0.95)*100:.1f}% correctly mapped to z={r.z_score:.4f}. "
                f"Reorder point ({r.reorder_point:.1f}) strictly exceeds mean lead time demand ({r.lead_time_demand_mean:.1f})."
            )

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            r: FixedPeriodReviewResult = result
            conv_str = f"\n- **Demand Normalization:** `{inputs['demand_conversion']}`" if inputs.get("demand_conversion") else ""
            exec_sum = (
                f"**Target Order-Up-To Level (S):** `{r.order_up_to_level:,.1f} units`."
                f"{conv_str}\n"
                f"- **Recommended Order Quantity Now (q):** `{r.recommended_order_quantity:,.1f} units`.\n"
                f"- **Review Interval:** Every `{r.review_period:.0f} {r.period_name}s` (Protection window: `{r.protection_interval:.0f} {r.period_name}s`).\n"
                f"- **Safety Stock Required:** `{r.safety_stock:,.1f} units`."
            )

            mean_d_val = inputs.get('avg_demand_per_period', r.inputs.get('demand_mean_per_period', 0.0))
            std_d_val = inputs.get('std_demand_per_period', r.inputs.get('demand_std_per_period', 0.0))
            formula_trace = [
                f"1. **Protection Interval:** $T_p = R + L = {r.review_period:.0f} + {r.lead_time:.0f} = {r.protection_interval:.0f}$ {r.period_name}s.",
                f"2. **Protection Demand Mean:** $\\mu_p = \\bar{{d}} (R + L) = {mean_d_val:.1f} \\times {r.protection_interval:.0f} = {r.protection_demand_mean:,.1f}$ units.",
                f"3. **Protection Demand Std Dev:** $\\sigma_p = \\sigma_d \\sqrt{{R + L}} = {std_d_val:.1f} \\times \\sqrt{{{r.protection_interval:.0f}}} = {r.protection_demand_std:,.2f}$ units.",
                f"4. **Safety Stock:** $SS = z \\times \\sigma_p = {r.z_score:.4f} \\times {r.protection_demand_std:,.2f} = {r.safety_stock:,.1f}$ units.",
                f"5. **Order-Up-To Level:** $S = \\mu_p + SS = {r.protection_demand_mean:,.1f} + {r.safety_stock:,.1f} = {r.order_up_to_level:,.1f}$ units.",
                f"6. **Order Quantity:** $q = \\max(0, S - IP) = \\max(0, {r.order_up_to_level:,.1f} - {r.current_inventory_position:,.1f}) = {r.recommended_order_quantity:,.1f}$ units.",
            ]

            insights = [
                f"**Periodic Vulnerability Window:** Because inventory is reviewed only every {r.review_period:.0f} {r.period_name}s, the system must buffer against demand uncertainty over {r.protection_interval:.0f} {r.period_name}s (R + L).",
                f"**Continuous vs Periodic Safety Stock:** Periodic review requires {r.comparison_vs_continuous_review_ss:.2f}x more safety stock than a continuous review system with the same lead time.",
                f"**Inventory Position:** Current stock on-hand ({inputs.get('current_on_hand', 0):,.0f}) plus on-order ({inputs.get('current_on_order', 0):,.0f}) equals {r.current_inventory_position:,.0f} units.",
            ]

            rec_action = (
                f"Place an order today for **{round(r.recommended_order_quantity):,d} units** "
                f"to bring inventory position up to the target base-stock level of **{round(r.order_up_to_level):,d} units**."
            )

            ver_sum = (
                f"Passed: Protection interval R+L verified. Safety stock ratio ({r.comparison_vs_continuous_review_ss:.2f}x) "
                f"matches theoretical expectation sqrt((R+L)/L). Non-negative order size confirmed."
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            r: NewsvendorResult = result
            exec_sum = (
                f"**Optimal Order Quantity (Q*):** `{r.optimal_order_quantity:,.1f} units`.\n"
                f"- **Critical Fractile (Service Level Target):** `{r.critical_fractile*100:.2f}%` ($C_u = \\${r.cost_of_underage:,.2f}$, $C_o = \\${r.cost_of_overage:,.2f}$).\n"
                f"- **Expected Profit:** `${r.expected_profit:,.2f}`.\n"
                f"- **Expected Sales:** `{r.expected_sales:,.1f} units` | **Expected Leftover (Waste):** `{r.expected_leftover_units:,.1f} units`."
            )

            p_disp = inputs.get('selling_price')
            c_disp = inputs.get('unit_cost')
            v_disp = inputs.get('salvage_value', 0.0)
            g_disp = inputs.get('stockout_penalty', 0.0)
            if p_disp is not None and c_disp is not None and p_disp > 0 and c_disp > 0:
                cu_str = f"$C_u = P - C + g =$ \\${p_disp:,.2f} - \\${c_disp:,.2f} + \\${g_disp:,.2f} = \\${r.cost_of_underage:,.2f}"
                co_str = f"$C_o = C - V =$ \\${c_disp:,.2f} - \\${v_disp:,.2f} = \\${r.cost_of_overage:,.2f}"
            else:
                cu_str = f"$C_u =$ \\${r.cost_of_underage:,.2f} (directly specified)"
                co_str = f"$C_o =$ \\${r.cost_of_overage:,.2f} (directly specified)"

            formula_trace = [
                f"1. **Cost of Underage (Lost Profit):** {cu_str}.",
                f"2. **Cost of Overage (Disposal Loss):** {co_str}.",
                f"3. **Critical Fractile:** $CR = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{r.cost_of_underage:,.2f}}}{{{r.cost_of_underage + r.cost_of_overage:,.2f}}} = {r.critical_fractile:.4f}$.",
                f"4. **Z-Score:** $z^* = \\Phi^{{-1}}({r.critical_fractile:.4f}) = {r.optimal_z_score:.4f}$.",
                f"5. **Optimal Quantity:** $Q^* = \\mu + z^* \\sigma = {inputs['mean_demand']:,.1f} + ({r.optimal_z_score:.4f} \\times {inputs['std_demand']:,.1f}) = {r.optimal_order_quantity:,.1f}$ units.",
                f"6. **Expected Profit:** $\\mathbb{{E}}[\\text{{Revenue}}] + \\mathbb{{E}}[\\text{{Salvage}}] - \\text{{Procurement Cost}} =$ \\${r.expected_profit:,.2f}.",
            ]

            bias_desc = (
                f"Because margin of lost sales ($C_u = \\${r.cost_of_underage:,.2f}$) exceeds markdown loss ($C_o = \\${r.cost_of_overage:,.2f}$), "
                f"the critical ratio ({r.critical_fractile*100:.1f}%) is above 50%, dictating that we stock {r.optimal_order_quantity - inputs['mean_demand']:,.1f} units above mean demand."
                if r.cost_of_underage > r.cost_of_overage
                else f"Because markdown loss ($C_o = \\${r.cost_of_overage:,.2f}$) exceeds unit margin ($C_u = \\${r.cost_of_underage:,.2f}$), "
                     f"the critical ratio is below 50%, dictating conservative ordering below mean demand."
            )

            insights = [
                f"**Risk Profile & Bias:** {bias_desc}",
                f"**Expected Waste vs Stockout:** Expected leftover unsold stock is {r.expected_leftover_units:,.1f} units (salvage value: ${r.expected_salvage_value:,.2f}), while expected lost sales are only {r.expected_lost_sales:,.1f} units.",
                f"**Value of Information:** The Expected Value of Perfect Information (EVPI) is ${r.expected_value_of_perfect_information:,.2f}. Improving forecasting accuracy can capture up to this amount in additional margin.",
            ]

            rec_action = (
                f"Procure **{round(r.optimal_order_quantity):,d} units** for the single selling period. "
                f"Establish liquidation channels to guarantee the ${inputs.get('salvage_value', 0):,.2f} salvage floor on leftover units."
            )

            ver_sum = (
                f"Passed: Critical fractile ({r.critical_fractile:.4f}) within bounds (0, 1). "
                f"Inventory balance holds: Expected Sales ({r.expected_sales:.1f}) + Expected Leftover ({r.expected_leftover_units:.1f}) = Q* ({r.optimal_order_quantity:.1f})."
            )

        else:
            exec_sum = "Unable to generate explanation for unknown model."
            formula_trace = []
            insights = []
            rec_action = "Please provide additional model details."
            ver_sum = "No verification performed."

        return AgentExplanation(
            executive_summary=exec_sum,
            formula_trace=formula_trace,
            managerial_insights=insights,
            verification_summary=ver_sum,
            recommended_action=rec_action,
        )

    # ========================================================================
    # MAIN PIPELINE: Process Customer Question
    # ========================================================================

    def process_query(self, question: str) -> AgentWorkflowResult:
        """
        Executes the full pipeline:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        # Step 1 & 2: Identify Model
        model_type, confidence, rationale = self.identify_model(question)

        if model_type == InventoryModelType.UNKNOWN:
            return AgentWorkflowResult(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs={},
                missing_inputs=["model_type", "parameters"],
                tool_selected="None",
                calculation_result=None,
                is_verified=False,
                verification_details={},
                explanation=None,
                error_message="Could not identify a supported inventory model from your query. Supported models: EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period Review, Newsvendor."
            )

        # Step 3: Extract Inputs
        extracted_inputs, missing = self.extract_inputs(question, model_type)

        if missing:
            return AgentWorkflowResult(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing,
                tool_selected="None",
                calculation_result=None,
                is_verified=False,
                verification_details={},
                explanation=None,
                error_message=f"Missing essential parameters for {model_type.value}: {', '.join(missing)}. Please include them in your query."
            )

        # Step 4 & 5: Select Python Tool & Calculate
        try:
            tool_name, calc_result = self.select_and_calculate_tool(model_type, extracted_inputs)
        except Exception as e:
            return AgentWorkflowResult(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                tool_selected="Error during tool invocation",
                calculation_result=None,
                is_verified=False,
                verification_details={"error": str(e)},
                explanation=None,
                error_message=f"Calculation error in {model_type.value}: {str(e)}"
            )

        # Step 6: Verify
        is_verified, verification_details = self.verify_result(model_type, calc_result)

        # Step 7: Explain
        explanation = self.explain_result(model_type, extracted_inputs, calc_result, is_verified)

        return AgentWorkflowResult(
            customer_question=question,
            identified_model=model_type,
            model_confidence=confidence,
            model_rationale=rationale,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            tool_selected=tool_name,
            calculation_result=calc_result,
            is_verified=is_verified,
            verification_details=verification_details,
            explanation=explanation,
            error_message=None,
            audit=getattr(calc_result, "audit", None),
        )
