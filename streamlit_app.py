"""
Supply Chain Analytics Inventory Decision Agent - Streamlit Web Application.

A modern, interactive web interface for natural-language inventory decision-making.
Connects directly to inventory_agent.py and inventory_tools.py.

Key Features:
- Natural-language inventory question input with 5 pre-loaded example problem buttons
- Automated model classification with confidence scoring & rationale
- Section 1: Extracted Parameters & Inferred Units
- Section 2: Mathematical Variable Definitions Glossary
- Section 3: Substituted Formulas with Step-by-Step LaTeX KaTeX Proofs
- Section 4: Intermediate Calculations, Feasibility & Optimality Verification Invariants
- Section 5: Executive Recommendations, KPI Metric Cards, Insights & Action Directives
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


# -----------------------------------------------------------------------------
# Pure-Python Pandas Compatibility Layer / Shim (when pandas is not installed)
# -----------------------------------------------------------------------------
class SimpleSeries:
    def __init__(self, data: Any, name: str = ""):
        self.data = list(data)
        self.name = name

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def sum(self) -> float:
        return sum(float(x) for x in self.data if x is not None and not (isinstance(x, float) and math.isnan(x)))

    def mean(self) -> float:
        valid = [float(x) for x in self.data if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(valid) / len(valid) if valid else 0.0

    def max(self) -> float:
        valid = [float(x) for x in self.data if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return max(valid) if valid else 0.0

    def min(self) -> float:
        valid = [float(x) for x in self.data if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return min(valid) if valid else 0.0

    def idxmax(self) -> int:
        best_v = float("-inf")
        best_idx = 0
        for i, val in enumerate(self.data):
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                fval = float(val)
                if fval > best_v:
                    best_v = fval
                    best_idx = i
        return best_idx

    def __mul__(self, other: Any) -> SimpleSeries:
        if isinstance(other, SimpleSeries):
            return SimpleSeries([float(a) * float(b) for a, b in zip(self.data, other.data)])
        elif isinstance(other, (int, float)):
            return SimpleSeries([float(a) * float(other) for a in self.data])
        return NotImplemented

    def fillna(self, val: Any) -> SimpleSeries:
        return SimpleSeries(
            [val if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in self.data],
            name=self.name,
        )

    def tolist(self) -> List[Any]:
        return list(self.data)


class SimpleDataFrame:
    def __init__(self, data: Any = None, columns: Optional[List[str]] = None):
        if data is None:
            self._data: Dict[str, List[Any]] = {}
            self._columns: List[str] = []
        elif isinstance(data, dict):
            self._data = {str(k): list(v) for k, v in data.items()}
            self._columns = list(columns) if columns is not None else list(self._data.keys())
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                cols = list(columns) if columns is not None else list(data[0].keys())
                self._columns = cols
                self._data = {c: [row.get(c, None) for row in data] for c in cols}
            elif data and isinstance(data[0], (list, tuple)):
                cols = list(columns) if columns is not None else [f"col_{i}" for i in range(len(data[0]))]
                self._columns = cols
                self._data = {c: [row[i] for row in data] for i, c in enumerate(cols)}
            else:
                self._data = {}
                self._columns = []
        else:
            self._data = {}
            self._columns = []

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @columns.setter
    def columns(self, new_cols: List[str]):
        if len(new_cols) != len(self._columns):
            raise ValueError("Length mismatch when setting DataFrame columns")
        new_data = {}
        for old_c, new_c in zip(self._columns, new_cols):
            new_data[new_c] = self._data[old_c]
        self._columns = list(new_cols)
        self._data = new_data

    @property
    def shape(self) -> Tuple[int, int]:
        n_rows = len(next(iter(self._data.values()))) if self._data else 0
        return (n_rows, len(self._columns))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, list):
            sub_data = {c: self._data[c] for c in key if c in self._data}
            return SimpleDataFrame(sub_data, columns=[c for c in key if c in self._data])
        if key in self._data:
            return SimpleSeries(self._data[key], name=str(key))
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, SimpleSeries):
            val_list = value.data
        elif isinstance(value, (list, tuple)):
            val_list = list(value)
        else:
            val_list = [value] * len(self)
        self._data[key] = val_list
        if key not in self._columns:
            self._columns.append(key)

    def copy(self) -> SimpleDataFrame:
        return SimpleDataFrame({k: list(v) for k, v in self._data.items()}, columns=list(self._columns))

    def rename(self, columns: Optional[Dict[str, str]] = None) -> SimpleDataFrame:
        if not columns:
            return self.copy()
        new_data = {}
        new_cols = []
        for c in self._columns:
            new_name = columns.get(c, c)
            new_data[new_name] = list(self._data[c])
            new_cols.append(new_name)
        return SimpleDataFrame(new_data, columns=new_cols)

    @property
    def loc(self) -> Any:
        class _LocIndexer:
            def __init__(self, parent: SimpleDataFrame):
                self.parent = parent

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                return {c: self.parent._data[c][idx] for c in self.parent._columns}

        return _LocIndexer(self)

    def to_dict(self, orient: str = "records") -> Any:
        if orient == "records":
            n_rows = len(self)
            return [{c: self._data[c][i] for c in self._columns} for i in range(n_rows)]
        return {k: list(v) for k, v in self._data.items()}

    def to_csv(self, index: bool = False) -> str:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=self._columns)
        writer.writeheader()
        writer.writerows(self.to_dict("records"))
        return out.getvalue()

    def sort_values(self, by: str, ascending: bool = True) -> SimpleDataFrame:
        records = self.to_dict("records")
        records.sort(key=lambda r: (r.get(by) is None, r.get(by)), reverse=not ascending)
        return SimpleDataFrame(records, columns=self._columns)

    def head(self, n: int = 5) -> SimpleDataFrame:
        records = self.to_dict("records")[:n]
        return SimpleDataFrame(records, columns=self._columns)

    def groupby(self, col: str) -> Any:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        n_rows = len(self)
        for i in range(n_rows):
            key = self._data[col][i]
            if key not in groups:
                groups[key] = []
            groups[key].append({c: self._data[c][i] for c in self._columns})

        class _GroupByObj:
            def __init__(self, grps: Dict[Any, List[Dict[str, Any]]]):
                self.grps = grps

            def agg(self, **kwargs: Any) -> SimpleDataFrame:
                records = []
                for g_val, rows in self.grps.items():
                    rec = {col: g_val}
                    for out_col, (src_col, func) in kwargs.items():
                        vals = [r[src_col] for r in rows if r.get(src_col) is not None]
                        if func == "sum":
                            rec[out_col] = sum(float(x) for x in vals)
                        elif func == "mean":
                            rec[out_col] = sum(float(x) for x in vals) / len(vals) if vals else 0.0
                        elif func == "count":
                            rec[out_col] = len(vals)
                        elif func == "max":
                            rec[out_col] = max(float(x) for x in vals) if vals else 0.0
                        elif func == "min":
                            rec[out_col] = min(float(x) for x in vals) if vals else 0.0
                    records.append(rec)
                return SimpleDataFrame(records)

        return _GroupByObj(groups)


class SimplePandasShim:
    DataFrame = SimpleDataFrame
    Series = SimpleSeries

    @staticmethod
    def read_csv(filepath_or_buffer: Any, *args: Any, **kwargs: Any) -> SimpleDataFrame:
        if hasattr(filepath_or_buffer, "getvalue"):
            raw = filepath_or_buffer.getvalue()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            f = io.StringIO(text)
        elif hasattr(filepath_or_buffer, "read"):
            raw = filepath_or_buffer.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            if hasattr(filepath_or_buffer, "seek"):
                filepath_or_buffer.seek(0)
            f = io.StringIO(text)
        elif isinstance(filepath_or_buffer, str):
            if "\n" in filepath_or_buffer or "\r" in filepath_or_buffer:
                f = io.StringIO(filepath_or_buffer)
            else:
                with open(filepath_or_buffer, "r", encoding="utf-8") as fp:
                    f = io.StringIO(fp.read())
        else:
            raise TypeError("Unsupported filepath_or_buffer type")

        reader = csv.DictReader(f)
        rows = list(reader)
        cols = list(reader.fieldnames) if reader.fieldnames else []
        return SimpleDataFrame(rows, columns=cols)

    @staticmethod
    def to_numeric(arg: Any, errors: str = "coerce") -> Any:
        def _conv(v: Any) -> float:
            if v is None:
                return 0.0
            try:
                s = str(v).replace("$", "").replace(",", "").strip()
                return float(s)
            except (ValueError, TypeError):
                if errors == "raise":
                    raise
                return 0.0

        if isinstance(arg, SimpleSeries):
            return SimpleSeries([_conv(x) for x in arg.data], name=arg.name)
        elif isinstance(arg, (list, tuple)):
            return SimpleSeries([_conv(x) for x in arg])
        return _conv(arg)


if pd is None:
    pd = SimplePandasShim()

try:
    import streamlit as st
except ModuleNotFoundError:
    # Graceful fallback mock for test inspection when streamlit is not installed in the active environment
    from unittest.mock import MagicMock
    st = MagicMock()

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    AgentWorkflowResult,
    AgentExplanation,
)
from inventory_tools import (
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    PeriodicReviewResult,
    NewsvendorResult,
    AnalystVerificationAudit,
)


# -----------------------------------------------------------------------------
# Module Constants: Variable Definitions Knowledge Base
# -----------------------------------------------------------------------------
VARIABLE_DEFINITIONS: Dict[InventoryModelType, List[Dict[str, str]]] = {
    InventoryModelType.EOQ: [
        {"Symbol": "D", "Name": "Annual Demand", "Definition": "Total customer demand rate per year", "Units": "units/year"},
        {"Symbol": "S", "Name": "Order / Setup Cost", "Definition": "Fixed cost incurred to place, transmit, and receive each replenishment order", "Units": "$/order"},
        {"Symbol": "H", "Name": "Holding / Carrying Cost", "Definition": "Cost to carry one unit in stock for one full year (capital cost, storage, insurance, spoilage)", "Units": "$/unit/year"},
        {"Symbol": "C", "Name": "Unit Purchase Price", "Definition": "Direct acquisition cost paid per unit to supplier", "Units": "$/unit"},
        {"Symbol": "L", "Name": "Lead Time", "Definition": "Elapsed time from purchase order placement until shipment arrival", "Units": "days"},
        {"Symbol": "Q*", "Name": "Economic Order Quantity", "Definition": "Batch size minimizing annual ordering cost + annual holding cost: sqrt(2DS/H)", "Units": "units/order"},
        {"Symbol": "ROP", "Name": "Reorder Point", "Definition": "Inventory level threshold that triggers replenishment placement: d * L", "Units": "units"},
        {"Symbol": "N", "Name": "Orders per Year", "Definition": "Replenishment order frequency: D / Q*", "Units": "orders/year"},
        {"Symbol": "T", "Name": "Order Cycle Interval", "Definition": "Average operational time between successive orders: Q* / D", "Units": "days/order"},
        {"Symbol": "AOC", "Name": "Annual Ordering Cost", "Definition": "Total annual cost spent placing orders: (D / Q*) * S", "Units": "$/year"},
        {"Symbol": "AHC", "Name": "Annual Holding Cost", "Definition": "Total annual cost spent carrying average cycle stock: (Q* / 2) * H", "Units": "$/year"},
        {"Symbol": "TC", "Name": "Total Annual Cost", "Definition": "Total procurement and inventory cost: D * C + AOC + AHC", "Units": "$/year"},
    ],
    InventoryModelType.QUANTITY_DISCOUNT: [
        {"Symbol": "D", "Name": "Annual Demand", "Definition": "Total annual unit demand required across the year", "Units": "units/year"},
        {"Symbol": "S", "Name": "Order / Setup Cost", "Definition": "Fixed ordering cost incurred per purchase order placed", "Units": "$/order"},
        {"Symbol": "i or r", "Name": "Annual Holding Rate", "Definition": "Inventory carrying charge expressed as a percentage fraction of purchase price", "Units": "% of price"},
        {"Symbol": "C_j", "Name": "Tier Unit Price", "Definition": "Unit purchase price applicable in discount quantity interval j", "Units": "$/unit"},
        {"Symbol": "H_j", "Name": "Tier Holding Cost", "Definition": "Holding cost per unit/year for tier j: i * C_j (or fixed holding cost)", "Units": "$/unit/year"},
        {"Symbol": "q_j", "Name": "Quantity Break", "Definition": "Minimum order quantity threshold required to qualify for discount tier price C_j", "Units": "units"},
        {"Symbol": "EOQ_j", "Name": "Raw EOQ", "Definition": "Unconstrained classic EOQ evaluated using holding cost H_j", "Units": "units"},
        {"Symbol": "Q_j*", "Name": "Feasible Order Quantity", "Definition": "Raw EOQ if within tier bounds; otherwise upward-adjusted to price break q_j", "Units": "units"},
        {"Symbol": "TC(Q_j*)", "Name": "Total Annual Tier Cost", "Definition": "Annual purchase spend (D * C_j) + Annual Ordering Cost + Annual Holding Cost", "Units": "$/year"},
        {"Symbol": "Savings", "Name": "Cost Reduction", "Definition": "Dollar savings achieved by taking discount vs base tier pricing", "Units": "$ and %"},
    ],
    InventoryModelType.SAFETY_STOCK_ROP: [
        {"Symbol": "d̄", "Name": "Average Demand", "Definition": "Expected customer demand per operational time period", "Units": "units/period"},
        {"Symbol": "σ_d", "Name": "Demand Standard Deviation", "Definition": "Statistical volatility / variability of customer demand per period", "Units": "units/period"},
        {"Symbol": "L̄", "Name": "Average Lead Time", "Definition": "Expected supplier replenishment transit duration", "Units": "periods"},
        {"Symbol": "σ_L", "Name": "Lead Time Std Dev", "Definition": "Statistical volatility / variability of supplier delivery lead time", "Units": "periods"},
        {"Symbol": "CSL", "Name": "Cycle Service Level", "Definition": "Target probability of satisfying all demand during replenishment lead time without stocking out", "Units": "% (0 to 1)"},
        {"Symbol": "z", "Name": "Safety Factor (Z-Score)", "Definition": "Standard normal inverse quantile corresponding to target CSL: Φ⁻¹(CSL)", "Units": "dimensionless"},
        {"Symbol": "σ_DLT", "Name": "Lead Time Demand Std Dev", "Definition": "Combined standard deviation of demand during lead time: sqrt(L̄ σ_d² + d̄² σ_L²)", "Units": "units"},
        {"Symbol": "SS", "Name": "Safety Stock", "Definition": "Buffer stock protecting against demand surges and supplier transit delays: z * σ_DLT", "Units": "units"},
        {"Symbol": "ROP", "Name": "Reorder Point", "Definition": "Inventory level threshold triggering a replenishment order: d̄ * L̄ + SS", "Units": "units"},
        {"Symbol": "E[Short]", "Name": "Expected Units Short", "Definition": "Expected unfilled units per replenishment cycle: σ_DLT * L(z)", "Units": "units/cycle"},
    ],
    InventoryModelType.FIXED_PERIOD_REVIEW: [
        {"Symbol": "R", "Name": "Review Period", "Definition": "Fixed interval elapsed between successive inventory status reviews and reorders", "Units": "periods"},
        {"Symbol": "L", "Name": "Replenishment Lead Time", "Definition": "Delivery transit time between review order placement and delivery arrival", "Units": "periods"},
        {"Symbol": "T_p", "Name": "Protection Interval", "Definition": "Vulnerability window across which inventory must cover demand: R + L", "Units": "periods"},
        {"Symbol": "d̄", "Name": "Average Demand", "Definition": "Expected customer demand per period", "Units": "units/period"},
        {"Symbol": "σ_d", "Name": "Demand Standard Deviation", "Definition": "Volatility of demand per period", "Units": "units/period"},
        {"Symbol": "μ_p", "Name": "Protection Demand Mean", "Definition": "Expected demand accumulated across the protection window: d̄ * (R + L)", "Units": "units"},
        {"Symbol": "σ_p", "Name": "Protection Demand Std Dev", "Definition": "Standard deviation of demand across protection window: σ_d * sqrt(R + L)", "Units": "units"},
        {"Symbol": "SS", "Name": "Periodic Safety Stock", "Definition": "Buffer stock held across R + L: z * σ_d * sqrt(R + L)", "Units": "units"},
        {"Symbol": "S", "Name": "Target Base-Stock Level", "Definition": "Order-up-to ceiling: Protection Demand Mean (μ_p) + Safety Stock (SS)", "Units": "units"},
        {"Symbol": "I_OH", "Name": "On-Hand Inventory", "Definition": "Physical stock currently stored in the warehouse / store", "Units": "units"},
        {"Symbol": "I_OO", "Name": "On-Order Inventory", "Definition": "Replenishment stock currently in transit or awaiting delivery", "Units": "units"},
        {"Symbol": "B", "Name": "Backorders", "Definition": "Unfilled customer orders awaiting stock replenishment", "Units": "units"},
        {"Symbol": "IP", "Name": "Inventory Position", "Definition": "Effective inventory currently controlled: I_OH + I_OO - B", "Units": "units"},
        {"Symbol": "q", "Name": "Order Quantity", "Definition": "Replenishment size ordered today: max(0, S - IP)", "Units": "units"},
    ],
    InventoryModelType.NEWSVENDOR: [
        {"Symbol": "P", "Name": "Selling / Retail Price", "Definition": "Revenue earned per unit sold at regular market price during the period", "Units": "$/unit"},
        {"Symbol": "C", "Name": "Unit Purchase Cost", "Definition": "Cost paid to procure or manufacture each unit before the period begins", "Units": "$/unit"},
        {"Symbol": "V", "Name": "Salvage Value", "Definition": "Discounted liquidation price received for leftover unsold units at season end", "Units": "$/unit"},
        {"Symbol": "g", "Name": "Stockout Penalty", "Definition": "Goodwill loss, loss of customer loyalty, or penalty per unmet unit", "Units": "$/unit"},
        {"Symbol": "C_u", "Name": "Cost of Underage", "Definition": "Opportunity loss from ordering one unit too few: P - C + g", "Units": "$/unit"},
        {"Symbol": "C_o", "Name": "Cost of Overage", "Definition": "Loss from ordering one unit too many that must be scrapped / salvaged: C - V", "Units": "$/unit"},
        {"Symbol": "CR", "Name": "Critical Ratio (Fractile)", "Definition": "Optimal service level balancing marginal underage vs overage costs: C_u / (C_u + C_o)", "Units": "fraction (0 to 1)"},
        {"Symbol": "μ", "Name": "Mean Demand", "Definition": "Expected customer demand for the single selling period", "Units": "units"},
        {"Symbol": "σ", "Name": "Demand Standard Deviation", "Definition": "Standard deviation of demand for the single selling period", "Units": "units"},
        {"Symbol": "z*", "Name": "Optimal Quantile", "Definition": "Standard normal inverse CDF at the critical ratio: Φ⁻¹(CR)", "Units": "dimensionless"},
        {"Symbol": "Q*", "Name": "Optimal Order Quantity", "Definition": "Single-period procurement quantity maximizing expected profit: μ + z* * σ", "Units": "units"},
        {"Symbol": "E[Sales]", "Name": "Expected Sales", "Definition": "Expected units sold to customers: μ - σ * L(z*)", "Units": "units"},
        {"Symbol": "E[Leftover]", "Name": "Expected Waste", "Definition": "Expected leftover units requiring salvage: Q* - E[Sales]", "Units": "units"},
        {"Symbol": "E[Profit]", "Name": "Expected Profit", "Definition": "Expected revenue + expected salvage minus procurement cost", "Units": "$"},
        {"Symbol": "EVPI", "Name": "Value of Perfect Info", "Definition": "Expected profit increase if customer demand were known in advance with 100% certainty", "Units": "$"},
    ],
}

# Example problem library
EXAMPLES = [
    {
        "label": "1. EOQ Model",
        "category": "Economic Order Quantity",
        "question": "Our annual demand is 12,000 units. Each order costs $50 to place, and holding cost is $2 per unit per year. Unit purchase price is $10. Lead time is 5 days. What is our optimal order quantity and reorder point?",
    },
    {
        "label": "2. Quantity Discount",
        "category": "All-Units Quantity Discount",
        "question": "A supplier offers volume discount: 1 to 499 units @ $10.00, 500 to 999 units @ $9.50, and 1000+ units @ $9.00. Annual demand is 4,000 units, order cost is $45, and annual holding rate is 20%. How much should we order to minimize total annual cost?",
    },
    {
        "label": "3. Safety Stock & ROP",
        "category": "Continuous Review (s, Q)",
        "question": "Daily demand averages 100 units with standard deviation of 15 units. Supplier lead time is 4 days with lead-time standard deviation of 1 day. We need a 95% cycle service level. What should our safety stock and reorder point be?",
    },
    {
        "label": "4. Periodic Review",
        "category": "Order-Up-To (R, S)",
        "question": "We review stock every 14 days with lead time of 7 days. Demand averages 50 units per day with std dev of 8. Target service level is 98%. On hand we have 420 units, 100 on order, and 20 backorders. What is the order-up-to level and how much should we order?",
    },
    {
        "label": "5. Newsvendor",
        "category": "Single-Period Stochastic",
        "question": "We sell winter jackets for $250 each. We buy them for $100. Any unsold jackets at the end of the season are sold at markdown for $40. Demand is normally distributed with mean 500 and std dev 80. How many jackets should we purchase?",
    },
]


# -----------------------------------------------------------------------------
# Formatting Helpers for Clean LaTeX & Invariant Table Display
# -----------------------------------------------------------------------------
def format_formula_for_display(formula_str: str) -> str:
    """
    Format a formula trace item for safe and clean Streamlit markdown and KaTeX rendering.
    - Moves currency dollar signs (e.g. \\$774.60) outside inline LaTeX $...$ blocks
      so Steps 3, 4, and 5 render cleanly without raw \\frac or \\text syntax.
    - Ensures currency figures outside LaTeX blocks are escaped (\\$...) so Streamlit's
      markdown parser does not confuse them with math delimiters.
    """
    f = str(formula_str)
    # Process paired unescaped $ delimiters
    delims = [m.start() for m in re.finditer(r"(?<!\\)\$", f)]
    if len(delims) >= 2 and len(delims) % 2 == 0:
        # Loop backwards over pairs so character indices remain valid
        for i in range(len(delims) - 2, -1, -2):
            start_idx = delims[i]
            end_idx = delims[i + 1]
            math_content = f[start_idx + 1 : end_idx]

            # Check if this math block contains currency figures like \$50 or $50
            curr_match = re.search(r"\\?\$[0-9]", math_content)
            if curr_match:
                c_pos = curr_match.start()
                math_prefix = math_content[:c_pos].rstrip()
                currency_suffix = math_content[c_pos:].strip()
                # Ensure all dollar signs in currency_suffix are escaped
                currency_suffix = re.sub(r"(?<!\\)\$", r"\$", currency_suffix)

                before = f[:start_idx]
                after = f[end_idx + 1:]
                if math_prefix:
                    f = f"{before}${math_prefix}$ {currency_suffix}{after}"
                else:
                    f = f"{before}{currency_suffix}{after}"

    # Escape any remaining unescaped currency dollar signs outside LaTeX blocks
    f = re.sub(r"(?<!\\)\$(?=[0-9])", r"\$", f)
    return f


def build_verification_table(ver_details: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Format invariant optimality checks and verification details for UI display.
    - Only evaluates boolean check fields (is_valid, holding_equals_ordering,
      second_derivative_strictly_positive, etc.) as '✓ PASS' / '✗ FAIL'.
    - Displays numeric difference and metric fields (cost_difference, relative_cost_difference,
      safety_stock_ratio_vs_continuous, etc.) as formatted numbers instead of '✗ FAIL'.
    """
    ver_table = []
    for check_name, val in ver_details.items():
        if check_name in ["sensitivity", "evaluated_tiers"]:
            continue
        if isinstance(val, bool):
            status_str = "✓ PASS" if val else "✗ FAIL"
        elif isinstance(val, float):
            status_str = f"{val:.6f}" if abs(val) < 1e-4 and val != 0 else f"{val:,.4f}"
        elif isinstance(val, int):
            status_str = f"{val:,d}"
        else:
            status_str = str(val)
        ver_table.append(
            {
                "Verification Invariant": check_name,
                "Status / Value": status_str,
            }
        )
    return ver_table


# -----------------------------------------------------------------------------
# Enterprise CSV Inventory Analytics Engine
# -----------------------------------------------------------------------------
SAMPLE_INVENTORY_CSV = """product,category,units,unit_cost
Widget Alpha,Electronics,500,12.50
Widget Beta,Electronics,300,24.00
Sensor Pro-100,Electronics,150,85.00
Display Screen 4K,Electronics,200,65.00
Gizmo Standard,Hardware,800,5.75
Gizmo Heavy-Duty,Hardware,1200,3.20
Steel Bolt M8 Pack,Fasteners,2500,0.45
Brass Hex Nut Pack,Fasteners,3000,0.30
Industrial Motor 5HP,Machinery,40,450.00
PLC Control Unit,Machinery,75,220.00
"""


def normalize_column_name(col: str) -> str:
    """
    Normalize column header string case-insensitively and map common inventory synonyms
    to standard canonical attributes: product, category, units, unit_cost.
    """
    cleaned = re.sub(r"[_\s\-]+", "_", col.strip().lower())
    if cleaned in ["product", "product_name", "item", "item_name", "sku", "name", "part", "part_name"]:
        return "product"
    if cleaned in ["category", "cat", "department", "dept", "group", "class", "product_category"]:
        return "category"
    if cleaned in ["units", "unit", "quantity", "qty", "stock", "on_hand", "inventory_units", "units_in_stock", "count"]:
        return "units"
    if cleaned in ["unit_cost", "cost", "unit_price", "price", "cost_per_unit", "unitcost", "cost_unit"]:
        return "unit_cost"
    return cleaned


def load_and_process_inventory_csv(file_or_content: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load an inventory CSV into a DataFrame, validate expected columns case-insensitively,
    and compute the inventory_value column (units * unit_cost).

    Returns:
        (processed_df, original_df, dataset_summary)
    """
    try:
        if hasattr(file_or_content, "getvalue"):
            raw_bytes = file_or_content.getvalue()
            raw_str = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else raw_bytes
            original_df = pd.read_csv(io.StringIO(raw_str))
        elif hasattr(file_or_content, "read"):
            raw = file_or_content.read()
            raw_str = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            if hasattr(file_or_content, "seek"):
                file_or_content.seek(0)
            original_df = pd.read_csv(io.StringIO(raw_str))
        elif isinstance(file_or_content, str):
            if "\n" in file_or_content or "\r" in file_or_content:
                original_df = pd.read_csv(io.StringIO(file_or_content))
            else:
                original_df = pd.read_csv(file_or_content)
        elif hasattr(file_or_content, "copy"):
            original_df = file_or_content.copy()
        else:
            original_df = pd.DataFrame(file_or_content)
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset: {e}")

    if original_df is None or len(original_df) == 0:
        raise ValueError("The uploaded CSV dataset is empty.")

    raw_columns = list(original_df.columns)
    col_mapping = {}
    for c in raw_columns:
        norm = normalize_column_name(str(c))
        if norm in ["product", "category", "units", "unit_cost"] and norm not in col_mapping.values():
            col_mapping[c] = norm

    required_targets = ["product", "category", "units", "unit_cost"]
    missing = [target for target in required_targets if target not in col_mapping.values()]
    if missing:
        raise ValueError(
            f"Missing required inventory column(s): {', '.join(missing)}. "
            f"Expected columns: product, category, units, unit_cost (handled case-insensitively). "
            f"Columns detected in CSV: {', '.join(raw_columns)}."
        )

    processed_df = original_df.copy()
    if hasattr(processed_df, "rename"):
        processed_df = processed_df.rename(columns=col_mapping)
    else:
        for old_col, new_col in col_mapping.items():
            processed_df[new_col] = processed_df[old_col]

    # Clean and convert numeric columns
    processed_df["units"] = pd.to_numeric(processed_df["units"], errors="coerce")
    if hasattr(processed_df["units"], "fillna"):
        processed_df["units"] = processed_df["units"].fillna(0)

    processed_df["unit_cost"] = pd.to_numeric(processed_df["unit_cost"], errors="coerce")
    if hasattr(processed_df["unit_cost"], "fillna"):
        processed_df["unit_cost"] = processed_df["unit_cost"].fillna(0.0)

    # Calculate inventory_value = units * unit_cost
    processed_df["inventory_value"] = processed_df["units"] * processed_df["unit_cost"]

    summary = {
        "num_rows": int(len(original_df)),
        "num_columns": int(len(raw_columns)),
        "column_names": list(raw_columns),
        "mapped_columns": col_mapping,
    }

    return processed_df, original_df, summary


def compute_inventory_metrics(df: Any) -> Dict[str, Any]:
    """
    Computes key dashboard metrics across all inventory products:
    - total units
    - average unit cost
    - product with the highest number of units
    - highest unit cost and corresponding product
    - total inventory value across all products
    """
    if hasattr(df, "to_dict"):
        records = df.to_dict("records")
    else:
        records = list(df)

    if not records:
        return {
            "total_units": 0.0,
            "average_unit_cost": 0.0,
            "weighted_avg_unit_cost": 0.0,
            "product_highest_units": "N/A",
            "highest_units_count": 0.0,
            "highest_unit_cost": 0.0,
            "product_highest_cost": "N/A",
            "total_inventory_value": 0.0,
            "total_products_count": 0,
        }

    total_units = 0.0
    total_inventory_value = 0.0
    unit_costs = []

    max_units = float("-inf")
    product_highest_units = "N/A"

    max_cost = float("-inf")
    product_highest_cost = "N/A"

    for r in records:
        prod_name = str(r.get("product", "Unknown")).strip()
        units = float(r.get("units", 0.0) or 0.0)
        cost = float(r.get("unit_cost", 0.0) or 0.0)
        val = float(r.get("inventory_value", units * cost) or 0.0)

        total_units += units
        total_inventory_value += val
        unit_costs.append(cost)

        if units > max_units:
            max_units = units
            product_highest_units = prod_name

        if cost > max_cost:
            max_cost = cost
            product_highest_cost = prod_name

    avg_unit_cost = sum(unit_costs) / len(unit_costs) if unit_costs else 0.0
    weighted_avg_unit_cost = (total_inventory_value / total_units) if total_units > 0 else 0.0

    return {
        "total_units": total_units,
        "average_unit_cost": avg_unit_cost,
        "weighted_avg_unit_cost": weighted_avg_unit_cost,
        "product_highest_units": product_highest_units,
        "highest_units_count": max(0.0, max_units) if max_units != float("-inf") else 0.0,
        "highest_unit_cost": max(0.0, max_cost) if max_cost != float("-inf") else 0.0,
        "product_highest_cost": product_highest_cost,
        "total_inventory_value": total_inventory_value,
        "total_products_count": len(records),
    }


def compute_category_breakdown(df: Any) -> List[Dict[str, Any]]:
    """
    Category-level breakdown grouping total units and total inventory value by category.
    Returns sorted list of dictionaries with product count, units, valuation, and shares.
    """
    if hasattr(df, "to_dict"):
        records = df.to_dict("records")
    else:
        records = list(df)

    if not records:
        return []

    cat_map: Dict[str, Dict[str, Any]] = {}
    grand_total_units = 0.0
    grand_total_value = 0.0

    for r in records:
        cat = str(r.get("category", "Uncategorized")).strip() or "Uncategorized"
        units = float(r.get("units", 0.0) or 0.0)
        cost = float(r.get("unit_cost", 0.0) or 0.0)
        val = float(r.get("inventory_value", units * cost) or 0.0)

        grand_total_units += units
        grand_total_value += val

        if cat not in cat_map:
            cat_map[cat] = {
                "category": cat,
                "product_count": 0,
                "total_units": 0.0,
                "total_inventory_value": 0.0,
                "cost_sum": 0.0,
            }
        cat_map[cat]["product_count"] += 1
        cat_map[cat]["total_units"] += units
        cat_map[cat]["total_inventory_value"] += val
        cat_map[cat]["cost_sum"] += cost

    breakdown = []
    for cat, info in cat_map.items():
        cnt = info["product_count"]
        t_units = info["total_units"]
        t_val = info["total_inventory_value"]
        avg_cost = info["cost_sum"] / cnt if cnt > 0 else 0.0
        pct_units = (t_units / grand_total_units * 100.0) if grand_total_units > 0 else 0.0
        pct_val = (t_val / grand_total_value * 100.0) if grand_total_value > 0 else 0.0

        breakdown.append({
            "category": cat,
            "product_count": cnt,
            "total_units": t_units,
            "pct_of_total_units": pct_units,
            "total_inventory_value": t_val,
            "pct_of_total_value": pct_val,
            "avg_unit_cost": avg_cost,
        })

    breakdown.sort(key=lambda x: x["total_inventory_value"], reverse=True)
    return breakdown


def render_csv_analytics_dashboard():
    """
    Render the CSV Inventory Data Upload & Analytics Dashboard.
    Displays:
    - Dataset summary (rows, columns, column names) & original data table
    - Key dashboard metrics: total units, average unit cost, product with highest units,
      highest unit cost, and total inventory value across all products.
    - Enriched data table with calculated inventory_value (units * unit_cost).
    - Category-level breakdown grouping total units and total inventory value by category.
    """
    st.markdown('<div class="section-header">📊 CSV Inventory Data Upload & Analytics Dashboard</div>', unsafe_allow_html=True)
    st.write(
        "Upload your company's inventory dataset as a CSV file. The analytics engine validates expected columns "
        "(`product`, `category`, `units`, and `unit_cost`, handled case-insensitively), calculates total "
        "holding valuation per product (`units * unit_cost`), surfaces aggregate KPI metrics, and generates category-level breakdowns."
    )

    col_up1, col_up2 = st.columns([3, 1.2])
    with col_up1:
        uploaded_file = st.file_uploader(
            "📂 Choose an Inventory CSV file:",
            type=["csv"],
            help="Expected columns: product, category, units, unit_cost (case-insensitive)",
            key="csv_file_uploader",
        )
    with col_up2:
        st.markdown("**Quick Demo:**")
        if st.button("📥 Load Sample CSV Data", help="Load sample 10-product catalog across 4 categories", use_container_width=True):
            st.session_state["use_sample_csv"] = True

    csv_data_source = None
    if uploaded_file is not None:
        csv_data_source = uploaded_file
        st.session_state["use_sample_csv"] = False
    elif st.session_state.get("use_sample_csv", False):
        csv_data_source = io.StringIO(SAMPLE_INVENTORY_CSV)

    if csv_data_source is None:
        st.info("👆 Please upload a CSV file or click '**📥 Load Sample CSV Data**' to view the analytics dashboard.")
        with st.expander("ℹ️ Expected CSV Format & Column Synonyms Guide", expanded=True):
            st.markdown(
                """
                The uploaded CSV can use standard column names or common synonyms (matching is case-insensitive):
                | Target Attribute | Supported Column Names (Case-Insensitive) | Description | Example |
                | :--- | :--- | :--- | :--- |
                | **product** | `product`, `product_name`, `item`, `sku`, `name`, `part` | Product SKU or identifier | `Widget Alpha` |
                | **category** | `category`, `department`, `dept`, `group`, `class` | Product category or department | `Electronics` |
                | **units** | `units`, `quantity`, `qty`, `stock`, `on_hand`, `count` | Physical units on-hand | `500` |
                | **unit_cost** | `unit_cost`, `cost`, `unit_price`, `price`, `cost_per_unit` | Unit procurement cost | `$12.50` |
                """
            )
        return

    try:
        processed_df, original_df, summary = load_and_process_inventory_csv(csv_data_source)
        metrics = compute_inventory_metrics(processed_df)
        cat_breakdown = compute_category_breakdown(processed_df)
    except Exception as e:
        st.error(f"❌ **CSV Processing Error:** {str(e)}")
        return

    # 1. Dataset Summary & Original Data Table
    st.markdown("---")
    st.markdown("#### 1. 📋 Dataset Summary & Uploaded Data Table")

    sum_cols = st.columns(3)
    with sum_cols[0]:
        st.metric("Total Rows (Products)", f"{summary['num_rows']:,d}")
    with sum_cols[1]:
        st.metric("Total Columns", f"{summary['num_columns']:,d}")
    with sum_cols[2]:
        mapped_count = len(summary["mapped_columns"])
        st.metric("Required Columns", f"✓ {mapped_count} / 4 Mapped")

    st.markdown(
        f"**Original Column Names ({summary['num_columns']}):** " + ", ".join([f"`{col}`" for col in summary["column_names"]])
    )

    with st.expander("📄 View Original Uploaded Data Table", expanded=False):
        if hasattr(original_df, "to_dict"):
            st.dataframe(original_df.to_dict("records"), use_container_width=True)
        else:
            st.dataframe(original_df, use_container_width=True)

    # 2. Key Dashboard Metrics
    st.markdown("---")
    st.markdown("#### 2. 📈 Key Inventory Dashboard Metrics")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric(
            label="Total Units",
            value=f"{metrics['total_units']:,.0f}",
            help="Total physical units across all products in inventory",
        )
    with kpi_cols[1]:
        st.metric(
            label="Average Unit Cost",
            value=f"${metrics['average_unit_cost']:,.2f}",
            help="Unweighted average procurement cost across all catalog SKUs",
        )
    with kpi_cols[2]:
        st.metric(
            label="Highest Stocked Product",
            value=f"{metrics['highest_units_count']:,.0f} units",
            delta=metrics["product_highest_units"],
            delta_color="normal",
            help="Product holding the highest physical unit volume",
        )
    with kpi_cols[3]:
        st.metric(
            label="Highest Unit Cost",
            value=f"${metrics['highest_unit_cost']:,.2f}",
            delta=metrics["product_highest_cost"],
            delta_color="normal",
            help="Highest single-unit purchase cost item",
        )
    with kpi_cols[4]:
        st.metric(
            label="Total Inventory Value",
            value=f"${metrics['total_inventory_value']:,.2f}",
            help="Total inventory asset valuation across all products: sum(units * unit_cost)",
        )

    # 3. Calculated inventory_value Column & Enriched Data Table
    st.markdown("---")
    st.markdown("#### 3. 💰 Enriched Inventory Data Table (`inventory_value = units * unit_cost`)")
    st.caption("Each line item displays the deterministic inventory valuation calculated from physical stock and unit cost:")

    table_records = []
    for r in processed_df.to_dict("records"):
        formatted_row = dict(r)
        formatted_row["units"] = f"{float(r['units']):,.0f}"
        formatted_row["unit_cost"] = f"${float(r['unit_cost']):,.2f}"
        formatted_row["inventory_value"] = f"${float(r['inventory_value']):,.2f}"
        table_records.append(formatted_row)

    st.dataframe(table_records, use_container_width=True)

    # Download button for enriched CSV
    enriched_csv_str = processed_df.to_csv(index=False) if hasattr(processed_df, "to_csv") else ""
    if enriched_csv_str:
        st.download_button(
            label="📥 Download Enriched Inventory CSV",
            data=enriched_csv_str,
            file_name="enriched_inventory_with_values.csv",
            mime="text/csv",
        )

    # 4. Category-Level Breakdown
    st.markdown("---")
    st.markdown("#### 4. 🏷️ Category-Level Breakdown")
    st.caption("Portfolio distribution grouping total units and total inventory value by product category:")

    cat_table_data = []
    for c in cat_breakdown:
        cat_table_data.append({
            "Category": c["category"],
            "Product Count": f"{c['product_count']:,d}",
            "Total Units": f"{c['total_units']:,.0f}",
            "Share of Units": f"{c['pct_of_total_units']:.1f}%",
            "Total Inventory Value": f"${c['total_inventory_value']:,.2f}",
            "Share of Value": f"{c['pct_of_total_value']:.1f}%",
            "Avg Unit Cost": f"${c['avg_unit_cost']:,.2f}",
        })
    st.table(cat_table_data)


# -----------------------------------------------------------------------------
# Main Application Renderer
# -----------------------------------------------------------------------------
def render_app():
    # -------------------------------------------------------------------------
    # 1. Page Configuration & Custom CSS
    # -------------------------------------------------------------------------
    st.set_page_config(
        page_title="Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1E40AF;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #E5E7EB;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }
        .model-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            font-weight: 700;
            border-radius: 9999px;
            background-color: #DBEAFE;
            color: #1E40AF;
            font-size: 0.95rem;
        }
        .verified-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            font-weight: 700;
            border-radius: 9999px;
            background-color: #D1FAE5;
            color: #065F46;
            font-size: 0.95rem;
        }
        .recommendation-card {
            background-color: #EFF6FF;
            border-left: 5px solid #2563EB;
            padding: 1rem 1.2rem;
            border-radius: 0.375rem;
            margin-bottom: 1rem;
        }
        .insight-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. Sidebar: Architecture & Guide
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/delivery-conveyer-belt.png", width=64)
        st.markdown("### Inventory Decision Agent")
        st.caption("Supply Chain Analytics & Optimization Engine")

        st.markdown("---")
        st.markdown("#### 🔄 Agent Architecture Flow")
        st.markdown(
            """
            1. **Question**: Natural-language supply chain inquiry
            2. **Identify Model**: Classifier selects best inventory policy
            3. **Extract Inputs**: High-precision terminology & entity extractor
            4. **Select Python Tool**: Dispatches to deterministic tool
            5. **Calculate**: Evaluates numerical formulas
            6. **Verify**: Tests mathematical optimality invariants
            7. **Explain**: Generates executive report & action items
            """
        )

        st.markdown("---")
        st.markdown("#### 📚 Supported Inventory Models")
        st.markdown(
            """
            - **EOQ**: Wilson formula trade-off
            - **Quantity Discount**: All-units cost minimization
            - **Safety Stock & ROP**: Buffer sizing under lead-time risk
            - **Fixed-Period Review**: Periodic base-stock $(R, S)$
            - **Newsvendor**: Perishable critical fractile
            """
        )

        st.markdown("---")
        st.caption("Powered by `inventory_agent.py` & `inventory_tools.py`")

    # -------------------------------------------------------------------------
    # 3. Header & Overview
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-title">📦 Supply Chain Analytics: Inventory Decision Agent & Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Solve single-item inventory problems with natural-language decision agents (EOQ, Quantity Discounts, '
        'Safety Stock & ROP, Periodic Review, Newsvendor) or upload enterprise inventory CSV datasets '
        'to automatically calculate valuation metrics, view executive KPI cards, and explore category distributions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Top-Level Mode Navigation: Decision Agent vs CSV Analytics Dashboard
    # -------------------------------------------------------------------------
    nav_tab_agent, nav_tab_dashboard = st.tabs([
        "💬 Natural-Language Decision Agent",
        "📊 CSV Inventory Upload & Analytics Dashboard",
    ])

    with nav_tab_dashboard:
        render_csv_analytics_dashboard()

    with nav_tab_agent:
        # 4. Example Problem Buttons
        # -------------------------------------------------------------------------
        st.markdown("##### 💡 Load an Example Problem:")
        example_cols = st.columns(len(EXAMPLES))

        for idx, ex in enumerate(EXAMPLES):
            with example_cols[idx]:
                if st.button(ex["label"], key=f"ex_btn_{idx}", use_container_width=True):
                    st.session_state["question_input"] = ex["question"]
                    st.session_state["auto_submit"] = True

        # -------------------------------------------------------------------------
        # 5. Natural Language Question Input
        # -------------------------------------------------------------------------
        default_val = st.session_state.get(
            "question_input",
            "Our annual demand is 12,000 units. Each order costs $50 to place, and holding cost is $2 per unit per year. Unit purchase price is $10. Lead time is 5 days. What is our optimal order quantity and reorder point?",
        )

        question_text = st.text_area(
            "Enter your inventory or supply chain question:",
            value=default_val,
            height=95,
            help="Type any inventory question. Mention annual demand, order/setup costs, holding rates, lead times, service levels, price breaks, or salvage values.",
        )

        col_run, col_clear, _ = st.columns([1.5, 1, 5])
        with col_run:
            run_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True)
        with col_clear:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state["question_input"] = ""
                st.session_state["auto_submit"] = False
                st.rerun()

        should_run = run_clicked or st.session_state.get("auto_submit", False)
        if st.session_state.get("auto_submit", False):
            st.session_state["auto_submit"] = False

        # -------------------------------------------------------------------------
        # 6. Agent Execution & Results Display
        # -------------------------------------------------------------------------
        if should_run:
            if not question_text.strip():
                st.warning("⚠️ Please enter a question or click an example button above to run the agent.")
            else:
                with st.spinner("Analyzing question, selecting model, and calculating optimal inventory parameters..."):
                    agent = InventoryDecisionAgent()
                    result: AgentWorkflowResult = agent.process_query(question_text)

                # -----------------------------------------------------------------
                # Header Badge & Model Classification Banner
                # -----------------------------------------------------------------
                st.markdown("---")
                badge_col1, badge_col2, badge_col3 = st.columns([3, 2, 2])

                with badge_col1:
                    st.markdown(
                        f'<span class="model-badge">Model Identified: {result.identified_model.value}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"**Classification Rationale:** {result.model_rationale}")

                with badge_col2:
                    conf_pct = int(result.model_confidence * 100)
                    st.metric(
                        label="Classification Confidence",
                        value=f"{conf_pct}%",
                        help="Keyword density and structural query matching confidence",
                    )

                with badge_col3:
                    if result.is_verified:
                        st.markdown('<span class="verified-badge">✓ Invariants Verified</span>', unsafe_allow_html=True)
                        st.caption(f"**Tool:** `{result.tool_selected}`")
                    elif result.missing_inputs:
                        st.markdown('<span style="color:#DC2626; font-weight:700;">⚠️ Incomplete Inputs</span>', unsafe_allow_html=True)
                        st.caption("Waiting for required inputs")
                    else:
                        st.markdown('<span style="color:#EA580C; font-weight:700;">⚠️ Verification Notice</span>', unsafe_allow_html=True)

                # -----------------------------------------------------------------
                # Handling Missing Inputs / Errors
                # -----------------------------------------------------------------
                if result.missing_inputs:
                    st.error(
                        f"**Missing Required Parameters for {result.identified_model.value}:**\n\n"
                        f"The agent identified this problem, but needs the following numerical parameters: "
                        f"`{', '.join(result.missing_inputs)}`.\n\n"
                        f"*Please revise your question to include these parameters.*"
                    )

                    # Show Variable Definitions to guide the user
                    st.markdown("##### 📖 Reference: Required Parameters & Definitions")
                    var_defs = VARIABLE_DEFINITIONS.get(result.identified_model, [])
                    st.table(var_defs)

                elif result.error_message:
                    st.error(f"**Execution Error:** {result.error_message}")

                elif result.calculation_result is not None and result.explanation is not None:
                    calc_res = result.calculation_result
                    expl = result.explanation
                    model_type = result.identified_model

                    # -------------------------------------------------------------
                    # KPI Metric Ribbon
                    # -------------------------------------------------------------
                    st.markdown("### 📊 Primary Quantitative Decision Metrics")
                    kpi_cols = st.columns(4)

                    if model_type == InventoryModelType.EOQ:
                        r: EOQResult = calc_res
                        with kpi_cols[0]:
                            st.metric("Optimal Order Quantity (Q*)", f"{r.eoq:,.1f} units", delta=f"{r.orders_per_year:.1f} orders/yr")
                        with kpi_cols[1]:
                            st.metric("Reorder Point (ROP)", f"{r.reorder_point:,.1f} units", delta=f"{r.order_interval_days:.1f} days cycle")
                        with kpi_cols[2]:
                            st.metric("Annual Inventory Cost", f"${r.total_annual_inventory_cost:,.2f}", help="Ordering Cost + Holding Cost")
                        with kpi_cols[3]:
                            st.metric("Total Annual Spend", f"${r.total_annual_cost:,.2f}", help="Purchase Spend + Inventory Cost")

                    elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
                        r: QuantityDiscountResult = calc_res
                        with kpi_cols[0]:
                            st.metric("Optimal Order Quantity", f"{r.optimal_order_qty:,.1f} units", delta=f"Tier {r.optimal_tier_index + 1}")
                        with kpi_cols[1]:
                            st.metric("Optimal Unit Price", f"${r.optimal_unit_price:,.2f} / unit", delta=f"-{r.percentage_savings:.1f}% vs base")
                        with kpi_cols[2]:
                            st.metric("Total Annual Cost", f"${r.optimal_total_annual_cost:,.2f}")
                        with kpi_cols[3]:
                            st.metric("Annual Savings", f"${r.dollar_savings:,.2f}", delta=f"{r.percentage_savings:.2f}% savings")

                    elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
                        r: SafetyStockROPResult = calc_res
                        with kpi_cols[0]:
                            st.metric("Safety Stock (SS)", f"{r.safety_stock:,.1f} units", delta=f"Buffer Stock")
                        with kpi_cols[1]:
                            st.metric("Reorder Point (ROP)", f"{r.reorder_point:,.1f} units", delta=f"Trigger Level")
                        with kpi_cols[2]:
                            csl_val = r.service_level if r.service_level is not None else 0.95
                            st.metric("Cycle Service Level", f"{csl_val * 100:.1f}%", delta=f"z = {r.z_score:.2f}")
                        with kpi_cols[3]:
                            st.metric("Stockout Risk per Cycle", f"{r.stockout_risk_per_cycle * 100:.2f}%", delta=f"{r.expected_short_per_cycle:.3f} units short", delta_color="inverse")

                    elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                        r: PeriodicReviewResult = calc_res
                        with kpi_cols[0]:
                            st.metric("Recommended Order (q)", f"{r.recommended_order_quantity:,.1f} units", delta="Order Now")
                        with kpi_cols[1]:
                            st.metric("Target Base-Stock (S)", f"{r.order_up_to_level:,.1f} units", delta=f"Order-Up-To Level")
                        with kpi_cols[2]:
                            st.metric("Protection Interval", f"{r.protection_interval:.0f} {r.period_name}s", delta=f"R={r.review_period:.0f} + L={r.lead_time:.0f}")
                        with kpi_cols[3]:
                            st.metric("Current Inventory Position", f"{r.current_inventory_position:,.0f} units", delta=f"SS: {r.safety_stock:,.1f} units")

                    elif model_type == InventoryModelType.NEWSVENDOR:
                        r: NewsvendorResult = calc_res
                        with kpi_cols[0]:
                            st.metric("Optimal Quantity (Q*)", f"{r.optimal_order_quantity:,.1f} units", delta=f"z* = {r.optimal_z_score:.2f}")
                        with kpi_cols[1]:
                            st.metric("Critical Fractile (CR)", f"{r.critical_fractile * 100:.2f}%", delta=f"Cu=${r.cost_of_underage:,.1f}, Co=${r.cost_of_overage:,.1f}")
                        with kpi_cols[2]:
                            st.metric("Expected Profit", f"${r.expected_profit:,.2f}")
                        with kpi_cols[3]:
                            st.metric("Expected Leftover", f"{r.expected_leftover_units:,.1f} units", delta=f"Sales: {r.expected_sales:,.1f} units")

                    # -------------------------------------------------------------
                    # Step-by-Step Analyst Verification Audit (Direct Python Output)
                    # -------------------------------------------------------------
                    audit_obj: Optional[AnalystVerificationAudit] = getattr(result, "audit", None) or getattr(calc_res, "audit", None)

                    if audit_obj is not None:
                        st.markdown("---")
                        st.markdown("### 🛡️ Step-by-Step Analyst Verification Audit (Direct Python Output)")
                        st.caption(
                            "Complete, deterministic mathematical audit computed directly by Python calculation tools. "
                            "All extracted parameters, definitions, formulas, substitutions, intermediate steps, and invariant verifications "
                            "are generated directly from Python calculations without language model alteration."
                        )

                        # Continuous readable audit block
                        with st.container():
                            st.markdown(
                                '<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid #2563EB; border-radius: 8px; padding: 22px; margin-bottom: 25px;">',
                                unsafe_allow_html=True,
                            )
                            st.markdown(audit_obj.to_markdown())
                            st.markdown('</div>', unsafe_allow_html=True)

                        with st.expander("📋 View Raw Continuous Audit Text / Export for Auditing"):
                            st.code(audit_obj.audit_text, language="markdown")

                    # -------------------------------------------------------------
                    # Interactive Deep-Dive Explorer & Verification Tabs
                    # -------------------------------------------------------------
                    st.markdown("---")
                    st.markdown("### 🔍 Interactive Deep-Dive Explorer & Detailed Tabs")
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(
                        [
                            "📋 Extracted Parameters",
                            "📖 Variable Definitions",
                            "📐 Substituted Formulas",
                            "⚙️ Intermediate Calculations",
                            "🏆 Final Recommendation",
                        ]
                    )

                    # =============================================================
                    # TAB 1: Extracted Parameters
                    # =============================================================
                    with tab1:
                        st.markdown('<div class="section-header">Section 1: Extracted Problem Parameters</div>', unsafe_allow_html=True)
                        st.write("Parameters automatically parsed from your question by the entity extraction engine:")

                        param_items = []
                        for k, v in result.extracted_inputs.items():
                            if k == "tiers":
                                continue
                            param_items.append({"Parameter": k, "Extracted Value": str(v)})

                        col_p1, col_p2 = st.columns([1.2, 1])
                        with col_p1:
                            st.table(param_items)

                        with col_p2:
                            if "tiers" in result.extracted_inputs:
                                st.markdown("**Extracted Quantity Discount Tiers:**")
                                tiers_display = []
                                for idx, t in enumerate(result.extracted_inputs["tiers"]):
                                    max_str = f"{int(t.max_qty):,d}" if t.max_qty is not None else "∞"
                                    tiers_display.append(
                                        {
                                            "Tier": t.tier_name if t.tier_name else f"Tier {idx+1}",
                                            "Min Qty": f"{int(t.min_qty):,d}",
                                            "Max Qty": max_str,
                                            "Unit Price": f"${t.unit_price:,.2f}",
                                        }
                                    )
                                st.table(tiers_display)
                            else:
                                st.info("No volume discount tiers required for this model.")

                    # =============================================================
                    # TAB 2: Variable Definitions
                    # =============================================================
                    with tab2:
                        st.markdown('<div class="section-header">Section 2: Mathematical Variable Definitions</div>', unsafe_allow_html=True)
                        st.write(f"Glossary of symbols, variables, and units specific to **{model_type.value}**:")

                        var_defs = VARIABLE_DEFINITIONS.get(model_type, [])
                        st.table(var_defs)

                    # =============================================================
                    # TAB 3: Substituted Formulas
                    # =============================================================
                    with tab3:
                        st.markdown('<div class="section-header">Section 3: Substituted Mathematical Formulas</div>', unsafe_allow_html=True)
                        st.write("Step-by-step mathematical derivations with extracted numerical values explicitly substituted:")

                        for f_idx, formula_item in enumerate(expl.formula_trace):
                            clean_formula = format_formula_for_display(formula_item)
                            st.markdown(f"**Step {f_idx + 1}:**")
                            st.markdown(clean_formula)

                    # =============================================================
                    # TAB 4: Intermediate Calculations & Verification
                    # =============================================================
                    with tab4:
                        st.markdown('<div class="section-header">Section 4: Intermediate Calculation Steps & Invariants</div>', unsafe_allow_html=True)

                        col_inter1, col_inter2 = st.columns([1.2, 1])

                        with col_inter1:
                            st.markdown("##### 🔢 Intermediate Numerical Values")
                            intermediates = getattr(calc_res, "intermediate_calculations", {})
                            inter_display = []
                            for k, v in intermediates.items():
                                if k in ["evaluated_tiers", "tier", "sensitivity"]:
                                    continue
                                if isinstance(v, float):
                                    val_str = f"{v:,.4f}"
                                else:
                                    val_str = str(v)
                                inter_display.append({"Intermediate Quantity": k, "Computed Value": val_str})
                            st.table(inter_display)

                        with col_inter2:
                            st.markdown("##### 🛡️ Invariant Optimality Checks")
                            ver_details = getattr(calc_res, "verification", {})
                            ver_table = build_verification_table(ver_details)
                            st.table(ver_table)

                        # Special table for Quantity Discount tier comparisons
                        if model_type == InventoryModelType.QUANTITY_DISCOUNT:
                            st.markdown("##### 🏷️ Quantity Discount Tier-by-Tier Evaluation Table")
                            qd_evals = getattr(calc_res, "tier_evaluations", [])
                            tier_table_data = []
                            for ev in qd_evals:
                                tier_table_data.append(
                                    {
                                        "Tier Name": ev.tier_name,
                                        "Unit Price": f"${ev.unit_price:,.2f}",
                                        "Holding Cost": f"${ev.holding_cost_unit:,.2f}",
                                        "Raw EOQ": f"{ev.raw_eoq:,.1f}",
                                        "Feasible Q": f"{ev.feasible_order_quantity:,.1f}",
                                        "Purchase Spend": f"${ev.annual_purchase_cost:,.2f}",
                                        "Order Cost": f"${ev.annual_ordering_cost:,.2f}",
                                        "Holding Cost": f"${ev.annual_holding_cost:,.2f}",
                                        "Total Annual Cost": f"${ev.total_annual_cost:,.2f}",
                                        "Optimal?": "★ BEST" if ev.feasible_order_quantity == getattr(calc_res, "optimal_order_qty", 0) else "",
                                    }
                                )
                            st.table(tier_table_data)

                    # =============================================================
                    # TAB 5: Final Recommendation & Action Plan
                    # =============================================================
                    with tab5:
                        st.markdown('<div class="section-header">Section 5: Final Executive Recommendation</div>', unsafe_allow_html=True)

                        # Executive Summary Box
                        st.markdown(
                            f'<div class="recommendation-card">'
                            f'<h4>Executive Decision Summary</h4>'
                            f'{expl.executive_summary}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Actionable Procurement Directive
                        st.success(f"**Recommended Action Directive:**\n\n{expl.recommended_action}")

                        # Managerial Insights & Trade-Offs
                        st.markdown("##### 💡 Managerial Insights & Trade-Offs")
                        for insight in expl.managerial_insights:
                            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

                        # Mathematical Assumptions
                        st.markdown("##### 📌 Operational & Mathematical Assumptions")
                        for assump in getattr(calc_res, "assumptions", []):
                            st.markdown(f"- *{assump}*")

                        # Verification summary
                        st.caption(f"**Verification Audit:** {expl.verification_summary}")

                    # -------------------------------------------------------------
                    # Downloadable JSON Report
                    # -------------------------------------------------------------
                    st.markdown("---")
                    col_down, _ = st.columns([2, 5])
                    with col_down:
                        json_payload = {
                            "question": result.customer_question,
                            "model": result.identified_model.value,
                            "confidence": result.model_confidence,
                            "inputs": {
                                k: (v if k != "tiers" else [vars(t) for t in v])
                                for k, v in result.extracted_inputs.items()
                            },
                            "result": calc_res.to_dict() if hasattr(calc_res, "to_dict") else str(calc_res),
                            "is_verified": result.is_verified,
                            "recommendation": expl.recommended_action,
                        }
                        st.download_button(
                            label="📥 Download JSON Decision Record",
                            data=json.dumps(json_payload, indent=2),
                            file_name="inventory_decision_report.json",
                            mime="application/json",
                            use_container_width=True,
                        )


if __name__ == "__main__" or "streamlit.runtime" in sys.modules:
    render_app()
