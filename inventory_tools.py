"""
Deterministic Supply Chain Analytics Inventory Tools.

Implements pure Python deterministic tools for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool adheres strictly to:
- Input validation (types, bounds, completeness)
- Structured results with typed dataclasses and to_dict()
- Intermediate calculations returned explicitly
- Final answer clearly exposed
- Operational & mathematical assumptions identified
- Never invents missing numerical inputs
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field, asdict
from statistics import NormalDist
from typing import List, Optional, Dict, Any, Union, Tuple


# ============================================================================
# Statistical Helper Utilities
# ============================================================================

def normal_standard_pdf(z: float) -> float:
    """Standard normal probability density function phi(z)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)


def normal_standard_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return NormalDist(0.0, 1.0).cdf(z)


def normal_standard_inv_cdf(p: float) -> float:
    """Standard normal quantile function Phi^{-1}(p)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability / service level must be strictly between 0 and 1. Got {p}")
    return NormalDist(0.0, 1.0).inv_cdf(p)


def normal_loss_function(z: float) -> float:
    """
    Standard normal unit loss function:
    L(z) = E[max(Z - z, 0)] = phi(z) - z * (1 - Phi(z))
    """
    phi = normal_standard_pdf(z)
    cdf = normal_standard_cdf(z)
    return phi - z * (1.0 - cdf)


# ============================================================================
# Structured Analyst Verification Audit
# ============================================================================

@dataclass
class AnalystVerificationAudit:
    """
    Structured Step-by-Step Analyst Verification Audit computed directly in Python.
    Guarantees mathematical integrity, input lineage, and deterministic reproducibility
    without language model hallucinations or altered numbers.
    """
    model_name: str
    extracted_inputs_and_units: List[Dict[str, str]]
    variable_definitions: List[Dict[str, str]]
    base_formula: str
    formula_with_substituted_values: str
    intermediate_calculations: List[Dict[str, Any]]
    final_verified_result: Dict[str, Any]
    audit_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return self.audit_text


def _format_audit_markdown(
    model_name: str,
    inputs_list: List[Dict[str, str]],
    var_defs: List[Dict[str, str]],
    base_formula: str,
    substituted_formula: str,
    intermediate_steps: List[Dict[str, Any]],
    final_result_summary: List[str],
    verification: Dict[str, Any],
) -> str:
    """Format the 6-stage Analyst Verification Audit as a continuous, readable Markdown document."""
    lines = []
    lines.append(f"### 🛡️ Step-by-Step Analyst Verification Audit: {model_name}")
    lines.append("*Direct Deterministic Python Calculation Output — Zero Language Model Alteration*\n")

    # 1. Extracted Inputs & Units
    lines.append("#### 1. Extracted Inputs & Units")
    lines.append("| Parameter | Symbol | Extracted Value | Operational Units |")
    lines.append("| :--- | :---: | :--- | :--- |")
    for row in inputs_list:
        lines.append(f"| {row.get('Parameter', '')} | {row.get('Symbol', '')} | {row.get('Value', '')} | {row.get('Units', '')} |")
    lines.append("")

    # 2. Variable Definitions
    lines.append("#### 2. Mathematical Variable Definitions")
    lines.append("| Symbol | Variable Name | Formal Mathematical Definition | Base Units |")
    lines.append("| :---: | :--- | :--- | :--- |")
    for row in var_defs:
        lines.append(f"| {row.get('Symbol', '')} | {row.get('Name', '')} | {row.get('Definition', '')} | {row.get('Units', '')} |")
    lines.append("")

    # 3. Base Formula
    lines.append("#### 3. Base Mathematical Formulas")
    lines.append(base_formula.strip())
    lines.append("")

    # 4. Formula with Substituted Values
    lines.append("#### 4. Substituted Mathematical Formulas")
    lines.append(substituted_formula.strip())
    lines.append("")

    # 5. Step-by-Step Intermediate Python Calculations
    lines.append("#### 5. Step-by-Step Intermediate Python Calculations")
    for step in intermediate_steps:
        st_num = step.get("step", "")
        st_name = step.get("name", "")
        st_form = step.get("formula", "")
        st_val = step.get("formatted_value", "")
        st_desc = step.get("description", "")
        lines.append(f"- **Step 5.{st_num} — {st_name}:** `{st_form}` = **{st_val}** ({st_desc})")
    lines.append("")

    # 6. Final Verified Result
    lines.append("#### 6. Final Verified Result & Invariant Checks")
    for item in final_result_summary:
        lines.append(item)
    lines.append("\n**Deterministic Invariant Verification Checks:**")
    lines.append("| Verification Invariant | Status / Value |")
    lines.append("| :--- | :--- |")
    for k, v in verification.items():
        if k in ["sensitivity", "evaluated_tiers"]:
            continue
        if isinstance(v, bool):
            stat = "✓ PASS" if v else "✗ FAIL"
        elif isinstance(v, float):
            stat = f"{v:.6f}" if abs(v) < 1e-4 and v != 0 else f"{v:,.4f}"
        elif isinstance(v, int):
            stat = f"{v:,d}"
        else:
            stat = str(v)
        lines.append(f"| `{k}` | **{stat}** |")
    lines.append("")

    return "\n".join(lines)


def _generate_eoq_audit(
    demand: float,
    setup_cost: float,
    hold_cost: float,
    item_cost: float,
    lead_time: float,
    work_days: int,
    q_star: float,
    orders_per_year: float,
    order_interval_days: float,
    annual_ordering_cost: float,
    annual_holding_cost: float,
    total_inventory_cost: float,
    annual_purchase_cost: float,
    total_cost: float,
    rop: float,
    verification: Dict[str, Any],
    demand_conversion_note: Optional[str] = None,
    holding_conversion_note: Optional[str] = None,
) -> AnalystVerificationAudit:
    num = 2.0 * demand * setup_cost
    ratio = num / hold_cost
    second_deriv = (2.0 * demand * setup_cost) / (q_star ** 3.0) if q_star > 0 else 0.0
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)

    demand_val_str = demand_conversion_note if demand_conversion_note else f"{demand:,.0f}"
    hold_val_str = holding_conversion_note if holding_conversion_note else f"\\${hold_cost:,.2f}"

    inputs_list = [
        {"Parameter": "Annual Demand", "Symbol": "$D$", "Value": demand_val_str, "Units": "units/year"},
        {"Parameter": "Order / Setup Cost", "Symbol": "$S$", "Value": f"\\${setup_cost:,.2f}", "Units": "$/order"},
        {"Parameter": "Holding / Carrying Cost", "Symbol": "$H$", "Value": hold_val_str, "Units": "$/unit/year"},
        {"Parameter": "Unit Purchase Price", "Symbol": "$C$", "Value": f"\\${item_cost:,.2f}" if item_cost > 0 else "Not specified (\\$0.00)", "Units": "$/unit"},
        {"Parameter": "Replenishment Lead Time", "Symbol": "$L$", "Value": f"{lead_time:.1f}" if lead_time > 0 else "0.0 (Instantaneous)", "Units": "days"},
        {"Parameter": "Annual Operating Days", "Symbol": "work_days", "Value": f"{work_days}", "Units": "days/year"},
    ]

    var_defs = [
        {"Symbol": "$D$", "Name": "Annual Demand", "Definition": "Total deterministic customer demand per annual cycle", "Units": "units/year"},
        {"Symbol": "$S$", "Name": "Order / Setup Cost", "Definition": "Fixed expense incurred to place and receive each replenishment order", "Units": "$/order"},
        {"Symbol": "$H$", "Name": "Holding / Carrying Cost", "Definition": "Annual cost to carry one unit in inventory for one full year", "Units": "$/unit/year"},
        {"Symbol": "$C$", "Name": "Unit Purchase Price", "Definition": "Direct unit acquisition price paid to supplier", "Units": "$/unit"},
        {"Symbol": "$Q^*$", "Name": "Economic Order Quantity", "Definition": "Cost-minimizing order batch size balancing ordering and holding costs", "Units": "units/order"},
        {"Symbol": "$N$", "Name": "Order Frequency", "Definition": "Number of replenishment orders placed per year ($D / Q^*$)", "Units": "orders/year"},
        {"Symbol": "$T$", "Name": "Order Cycle Duration", "Definition": "Average operational elapsed time between successive replenishment orders", "Units": "working days"},
        {"Symbol": "$\\text{AOC}$", "Name": "Annual Ordering Cost", "Definition": "Total yearly ordering expenditure: $(D / Q^*) \\times S$", "Units": "$/year"},
        {"Symbol": "$\\text{AHC}$", "Name": "Annual Holding Cost", "Definition": "Total yearly holding expenditure: $(Q^* / 2) \\times H$", "Units": "$/year"},
        {"Symbol": "$\\text{AIC}$", "Name": "Total Annual Inventory Cost", "Definition": "Total annual inventory-related cost: $\\text{AOC} + \\text{AHC}$", "Units": "$/year"},
        {"Symbol": "$\\text{ROP}$", "Name": "Reorder Point", "Definition": "Inventory level threshold that triggers placement of a replenishment order", "Units": "units"},
    ]

    base_formula = (
        "- **Economic Order Quantity (EOQ):** $Q^* = \\sqrt{\\frac{2DS}{H}}$\n"
        "- **Annual Order Frequency:** $N = \\frac{D}{Q^*}$\n"
        "- **Order Cycle Duration:** $T = \\left(\\frac{Q^*}{D}\\right) \\times \\text{work\\_days}$\n"
        "- **Annual Ordering Cost:** $\\text{AOC} = \\left(\\frac{D}{Q^*}\\right) S$\n"
        "- **Annual Holding Cost:** $\\text{AHC} = \\left(\\frac{Q^*}{2}\\right) H$\n"
        "- **Total Annual Inventory Cost:** $\\text{AIC} = \\text{AOC} + \\text{AHC}$\n"
        "- **Reorder Point (ROP):** $\\text{ROP} = \\left(\\frac{D}{\\text{work\\_days}}\\right) \\times L$"
    )

    substituted_formula = (
        f"- **Optimal Order Quantity (EOQ):** $Q^* = \\sqrt{{\\frac{{2 \\times {demand:,.0f} \\times {setup_cost:,.2f}}}{{{hold_cost:,.2f}}}}} = \\sqrt{{\\frac{{{num:,.2f}}}{{{hold_cost:,.2f}}}}} = \\sqrt{{{ratio:,.2f}}} = {q_star:,.2f}$ units\n"
        f"- **Annual Order Frequency:** $N = \\frac{{{demand:,.0f}}}{{{q_star:,.2f}}} = {orders_per_year:.2f}$ orders/year\n"
        f"- **Order Cycle Duration:** $T = \\left(\\frac{{{q_star:,.2f}}}{{{demand:,.0f}}}\\right) \\times {work_days} = {order_interval_days:.1f}$ working days\n"
        f"- **Annual Ordering Cost:** $\\text{{AOC}} = {orders_per_year:.2f} \\times$ \\${setup_cost:,.2f} = \\${annual_ordering_cost:,.2f}\n"
        f"- **Annual Holding Cost:** $\\text{{AHC}} = \\left(\\frac{{{q_star:,.2f}}}{{2}}\\right) \\times$ \\${hold_cost:,.2f} = \\${annual_holding_cost:,.2f}\n"
        f"- **Total Annual Inventory Cost:** $\\text{{AIC}} =$ \\${annual_ordering_cost:,.2f} + \\${annual_holding_cost:,.2f} = \\${total_inventory_cost:,.2f}\n"
        f"- **Reorder Point (ROP):** $\\text{{ROP}} = \\left(\\frac{{{demand:,.0f}}}{{{work_days}}}\\right) \\times {lead_time:.1f} = {rop:,.1f}$ units"
    )

    intermediate_steps = []
    step_num = 1
    if demand_conversion_note:
        intermediate_steps.append({
            "step": f"{step_num}",
            "name": "Demand Rate Annualization",
            "formula": demand_conversion_note,
            "value": demand,
            "formatted_value": f"{demand:,.0f} units/year",
            "description": "Normalized non-annual demand rate to standard 12-month annual horizon (D)",
        })
        step_num += 1

    if holding_conversion_note:
        intermediate_steps.append({
            "step": f"{step_num}",
            "name": "Holding Cost Annualization",
            "formula": holding_conversion_note,
            "value": hold_cost,
            "formatted_value": f"${hold_cost:,.2f}/unit/year",
            "description": "Normalized non-annual holding cost to standard 1-year annual horizon (H)",
        })
        step_num += 1

    intermediate_steps.extend([
        {"step": step_num, "name": "Ordering Numerator (2DS)", "formula": "2 * D * S", "value": num, "formatted_value": f"{num:,.4f}", "description": "Scale product of annual demand and fixed order setup cost"},
        {"step": step_num + 1, "name": "Holding Cost Ratio (2DS / H)", "formula": "(2 * D * S) / H", "value": ratio, "formatted_value": f"{ratio:,.4f}", "description": "Trade-off ratio between ordering scale and unit carrying cost"},
        {"step": step_num + 2, "name": "Unrounded Optimal Lot (Q*)", "formula": "sqrt(2DS / H)", "value": q_star, "formatted_value": f"{q_star:,.4f} units", "description": "Analytical cost-minimizing order batch size"},
        {"step": step_num + 3, "name": "Annual Orders (N)", "formula": "D / Q*", "value": orders_per_year, "formatted_value": f"{orders_per_year:,.4f} orders/yr", "description": "Replenishment frequency required across planning year"},
        {"step": step_num + 4, "name": "Order Interval in Days (T)", "formula": "(Q* / D) * work_days", "value": order_interval_days, "formatted_value": f"{order_interval_days:,.2f} days", "description": "Working days elapsed between consecutive replenishment deliveries"},
        {"step": step_num + 5, "name": "Annual Ordering Cost (AOC)", "formula": "N * S", "value": annual_ordering_cost, "formatted_value": f"${annual_ordering_cost:,.2f}", "description": "Total yearly expense spent placing replenishment orders"},
        {"step": step_num + 6, "name": "Annual Holding Cost (AHC)", "formula": "(Q* / 2) * H", "value": annual_holding_cost, "formatted_value": f"${annual_holding_cost:,.2f}", "description": "Total yearly expense spent carrying cycle stock"},
        {"step": step_num + 7, "name": "First-Order Cost Difference", "formula": "|AOC - AHC|", "value": cost_diff, "formatted_value": f"${cost_diff:,.4f}", "description": "Optimality balance: at true EOQ, ordering cost exactly balances holding cost"},
        {"step": step_num + 8, "name": "Second Derivative (Convexity)", "formula": "2DS / (Q*)^3", "value": second_deriv, "formatted_value": f"{second_deriv:.6f}", "description": "Strict convexity test confirming that Q* is a unique global minimum"},
    ])

    final_result = {
        "optimal_order_quantity": q_star,
        "reorder_point": rop,
        "orders_per_year": orders_per_year,
        "order_interval_days": order_interval_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_annual_inventory_cost": total_inventory_cost,
        "total_annual_cost": total_cost,
        "verification_invariants": verification,
        "is_valid": verification.get("is_valid", True),
    }

    final_summary = [
        f"- **Optimal Order Quantity ($Q^*$):** `{q_star:,.1f} units` per order",
        f"- **Reorder Point (ROP):** `{rop:,.1f} units` (Lead Time: `{lead_time:.1f} days`)",
        f"- **Annual Ordering Cost (AOC):** `\\${annual_ordering_cost:,.2f}`",
        f"- **Annual Holding Cost (AHC):** `\\${annual_holding_cost:,.2f}`",
        f"- **Total Annual Inventory Cost (AIC):** `\\${total_inventory_cost:,.2f}`",
        f"- **Total Annual Spend (including acquisition):** `\\${total_cost:,.2f}`",
        f"- **Order Frequency & Cycle:** `{orders_per_year:.2f} orders/year` (~every `{order_interval_days:.1f} working days`)",
    ]

    audit_md = _format_audit_markdown(
        model_name="Economic Order Quantity (EOQ)",
        inputs_list=inputs_list,
        var_defs=var_defs,
        base_formula=base_formula,
        substituted_formula=substituted_formula,
        intermediate_steps=intermediate_steps,
        final_result_summary=final_summary,
        verification=verification,
    )

    return AnalystVerificationAudit(
        model_name="Economic Order Quantity (EOQ)",
        extracted_inputs_and_units=inputs_list,
        variable_definitions=var_defs,
        base_formula=base_formula,
        formula_with_substituted_values=substituted_formula,
        intermediate_calculations=intermediate_steps,
        final_verified_result=final_result,
        audit_text=audit_md,
    )


def _generate_quantity_discount_audit(
    demand: float,
    setup_cost: float,
    h_rate: Optional[float],
    f_hold: Optional[float],
    tier_evaluations: List[TierEvaluationResult],
    best_idx: int,
    optimal_tier: TierEvaluationResult,
    base_cost: float,
    savings: float,
    pct_savings: float,
    verification: Dict[str, Any],
    demand_conversion_note: Optional[str] = None,
    holding_conversion_note: Optional[str] = None,
) -> AnalystVerificationAudit:
    demand_val_str = demand_conversion_note if demand_conversion_note else f"{demand:,.0f}"
    hold_desc = (
        holding_conversion_note
        if holding_conversion_note
        else (f"{h_rate*100:.1f}% of unit price" if h_rate is not None else f"\\${f_hold:,.2f} / unit / year")
    )

    inputs_list = [
        {"Parameter": "Annual Demand", "Symbol": "$D$", "Value": demand_val_str, "Units": "units/year"},
        {"Parameter": "Order / Setup Cost", "Symbol": "$S$", "Value": f"\\${setup_cost:,.2f}", "Units": "$/order"},
        {"Parameter": "Holding Cost Structure", "Symbol": "$H$", "Value": hold_desc, "Units": "% or $/unit/yr"},
        {"Parameter": "Total Discount Tiers", "Symbol": "tiers_count", "Value": f"{len(tier_evaluations)}", "Units": "tiers evaluated"},
    ]
    for idx, ev in enumerate(tier_evaluations):
        max_str = f"{int(ev.max_qty):,d}" if ev.max_qty is not None else "∞"
        inputs_list.append({
            "Parameter": f"Discount {ev.tier_name}",
            "Symbol": f"Tier {idx+1}",
            "Value": f"Range: [{int(ev.min_qty):,d} - {max_str}], Price: \\${ev.unit_price:,.2f}",
            "Units": "units, $/unit",
        })

    var_defs = [
        {"Symbol": "$D$", "Name": "Annual Demand", "Definition": "Total deterministic customer demand per year", "Units": "units/year"},
        {"Symbol": "$S$", "Name": "Order / Setup Cost", "Definition": "Fixed expense per replenishment purchase order", "Units": "$/order"},
        {"Symbol": "$C_j$", "Name": "Unit Price (Tier j)", "Definition": "Discounted purchase price in tier j", "Units": "$/unit"},
        {"Symbol": "$H_j$", "Name": "Holding Cost (Tier j)", "Definition": "Carrying cost per unit per year for tier j: i * C_j or fixed H", "Units": "$/unit/year"},
        {"Symbol": "$Q_j$", "Name": "Raw EOQ (Tier j)", "Definition": "Unconstrained EOQ calculated at tier price: sqrt(2DS / H_j)", "Units": "units"},
        {"Symbol": "$Q_j^*$", "Name": "Feasible Order Qty", "Definition": "Feasible batch size: raw EOQ if within tier bounds, or rounded up to price break", "Units": "units"},
        {"Symbol": "$\\text{APC}_j$", "Name": "Annual Purchase Cost", "Definition": "Annual direct procurement spend: D * C_j", "Units": "$/year"},
        {"Symbol": "$\\text{AOC}_j$", "Name": "Annual Ordering Cost", "Definition": "Annual order placement spend: (D / Q_j^*) * S", "Units": "$/year"},
        {"Symbol": "$\\text{AHC}_j$", "Name": "Annual Holding Cost", "Definition": "Annual inventory carrying spend: (Q_j^* / 2) * H_j", "Units": "$/year"},
        {"Symbol": "$\\text{TAC}_j$", "Name": "Total Annual Cost", "Definition": "Grand total annual spend: APC_j + AOC_j + AHC_j", "Units": "$/year"},
    ]

    base_formula = (
        "- **Raw EOQ per Tier:** $Q_j = \\sqrt{\\frac{2DS}{H_j}}$\n"
        "- **Feasibility Rule:** If $q_{j,\\min} \\le Q_j \\le q_{j,\\max}$, $Q_j^* = Q_j$; if $Q_j < q_{j,\\min}$, $Q_j^* = q_{j,\\min}$ (adjusted to price break); if $Q_j > q_{j,\\max}$, infeasible.\n"
        "- **Total Annual Cost per Tier:** $\\text{TAC}(Q_j^*) = D C_j + \\left(\\frac{D}{Q_j^*}\\right) S + \\left(\\frac{Q_j^*}{2}\\right) H_j$\n"
        "- **Global Optimal Order Lot:** $Q^* = \\arg\\min_j \\text{TAC}(Q_j^*)$"
    )

    sub_lines = []
    for idx, ev in enumerate(tier_evaluations):
        status_txt = "Feasible EOQ" if ev.is_raw_eoq_feasible else "Adjusted to Tier Price Break"
        sub_lines.append(
            f"- **Tier {idx+1} ({ev.tier_name} @ \\${ev.unit_price:,.2f}):** "
            f"Raw $Q_{idx+1} = \\sqrt{{\\frac{{2 \\times {demand:,.0f} \\times {setup_cost:,.2f}}}{{{ev.holding_cost_unit:,.2f}}}}} = {ev.raw_eoq:,.1f}\\text{{ units}} \\rightarrow$ "
            f"Feasible $Q_{idx+1}^* = {ev.feasible_order_quantity:,.1f}$ ({status_txt}) $\\rightarrow$ "
            f"$\\text{{TAC}} = \\${ev.annual_purchase_cost:,.2f} + \\${ev.annual_ordering_cost:,.2f} + \\${ev.annual_holding_cost:,.2f} = \\${ev.total_annual_cost:,.2f}$"
        )
    substituted_formula = "\n".join(sub_lines)

    intermediate_steps = []
    st_idx = 1
    for idx, ev in enumerate(tier_evaluations):
        intermediate_steps.append({
            "step": st_idx,
            "name": f"Tier {idx+1} Raw EOQ",
            "formula": f"sqrt(2 * {demand:,.0f} * {setup_cost:,.2f} / {ev.holding_cost_unit:,.2f})",
            "value": ev.raw_eoq,
            "formatted_value": f"{ev.raw_eoq:,.2f} units",
            "description": f"Unconstrained EOQ at price \\${ev.unit_price:,.2f}",
        })
        st_idx += 1
        status_desc = "Feasible EOQ within tier bounds" if ev.is_raw_eoq_feasible else "Adjusted up to tier price break minimum"
        intermediate_steps.append({
            "step": st_idx,
            "name": f"Tier {idx+1} Feasible Lot & Total Cost",
            "formula": f"Purchase(\\${ev.annual_purchase_cost:,.2f}) + Order(\\${ev.annual_ordering_cost:,.2f}) + Holding(\\${ev.annual_holding_cost:,.2f})",
            "value": ev.total_annual_cost,
            "formatted_value": f"${ev.total_annual_cost:,.2f}",
            "description": f"Feasible lot Q={ev.feasible_order_quantity:,.0f} ({status_desc})",
        })
        st_idx += 1

    intermediate_steps.append({
        "step": st_idx,
        "name": "Net Annual Cost Savings vs Base Tier",
        "formula": f"${base_cost:,.2f} - ${optimal_tier.total_annual_cost:,.2f}",
        "value": savings,
        "formatted_value": f"${savings:,.2f} ({pct_savings:.2f}% savings)",
        "description": "Total annual cost reduction realized by ordering at optimal price break",
    })

    final_result = {
        "optimal_order_quantity": optimal_tier.feasible_order_quantity,
        "optimal_tier_index": best_idx,
        "optimal_tier_name": optimal_tier.tier_name,
        "optimal_unit_price": optimal_tier.unit_price,
        "optimal_total_annual_cost": optimal_tier.total_annual_cost,
        "optimal_annual_purchase_cost": optimal_tier.annual_purchase_cost,
        "optimal_annual_ordering_cost": optimal_tier.annual_ordering_cost,
        "optimal_annual_holding_cost": optimal_tier.annual_holding_cost,
        "base_tier_cost": base_cost,
        "annual_savings_vs_base": savings,
        "percentage_savings_vs_base": pct_savings,
        "verification_invariants": verification,
        "is_valid": verification.get("is_valid", True),
    }

    final_summary = [
        f"- **Optimal Order Quantity ($Q^*$):** `{optimal_tier.feasible_order_quantity:,.1f} units` (Tier {best_idx+1}: {optimal_tier.tier_name})",
        f"- **Optimal Unit Price:** `\\${optimal_tier.unit_price:,.2f} / unit`",
        f"- **Total Annual Cost (Spend):** `\\${optimal_tier.total_annual_cost:,.2f}`",
        f"- **Cost Breakdown:** Purchase: `\\${optimal_tier.annual_purchase_cost:,.2f}` | Order: `\\${optimal_tier.annual_ordering_cost:,.2f}` | Holding: `\\${optimal_tier.annual_holding_cost:,.2f}`",
        f"- **Net Annual Savings vs Base Pricing:** `\\${savings:,.2f}` (`{pct_savings:.2f}%` cost reduction)",
    ]

    audit_md = _format_audit_markdown(
        model_name="All-Units Quantity Discount Model",
        inputs_list=inputs_list,
        var_defs=var_defs,
        base_formula=base_formula,
        substituted_formula=substituted_formula,
        intermediate_steps=intermediate_steps,
        final_result_summary=final_summary,
        verification=verification,
    )

    return AnalystVerificationAudit(
        model_name="All-Units Quantity Discount Model",
        extracted_inputs_and_units=inputs_list,
        variable_definitions=var_defs,
        base_formula=base_formula,
        formula_with_substituted_values=substituted_formula,
        intermediate_calculations=intermediate_steps,
        final_verified_result=final_result,
        audit_text=audit_md,
    )


def _generate_rop_audit(
    d_val: float,
    sig_d: float,
    L_val: float,
    sig_L_val: float,
    sl_val: Optional[float],
    calc_z: float,
    calc_csl: float,
    is_stochastic: bool,
    lead_time_demand_mean: float,
    sigma_LT: float,
    ss: float,
    rop: float,
    stockout_risk: float,
    loss_val: float,
    expected_short: float,
    period_str: str,
    case_name: str,
    verification: Dict[str, Any],
    demand_conversion_note: Optional[str] = None,
    lead_time_conversion_note: Optional[str] = None,
) -> AnalystVerificationAudit:
    d_val_str = f"{d_val:,.1f}" + (f" ({demand_conversion_note})" if demand_conversion_note else "")
    L_val_str = f"{L_val:.1f}" + (f" ({lead_time_conversion_note})" if lead_time_conversion_note else "")
    inputs_list = [
        {"Parameter": "Average Demand per Period", "Symbol": "$\\bar{d}$", "Value": d_val_str, "Units": f"units/{period_str}"},
        {"Parameter": "Demand Std Dev per Period", "Symbol": "$\\sigma_d$", "Value": f"{sig_d:,.2f}", "Units": f"units/{period_str}"},
        {"Parameter": "Average Lead Time", "Symbol": "$\\bar{L}$", "Value": L_val_str, "Units": f"{period_str}s"},
        {"Parameter": "Lead Time Std Dev", "Symbol": "$\\sigma_L$", "Value": f"{sig_L_val:.2f}" if sig_L_val > 0 else "0.0 (Deterministic)", "Units": f"{period_str}s"},
        {"Parameter": "Cycle Service Level", "Symbol": "$\\text{CSL}$", "Value": f"{calc_csl*100:.1f}%", "Units": "% in-stock probability"},
        {"Parameter": "Operational Case", "Symbol": "case_type", "Value": case_name, "Units": "model taxonomy"},
    ]

    var_defs = [
        {"Symbol": "$\\bar{d}$", "Name": "Mean Period Demand", "Definition": "Average customer demand rate per planning period", "Units": f"units/{period_str}"},
        {"Symbol": "$\\sigma_d$", "Name": "Period Demand Std Dev", "Definition": "Standard deviation of customer demand per planning period", "Units": f"units/{period_str}"},
        {"Symbol": "$\\bar{L}$", "Name": "Mean Lead Time", "Definition": "Average supplier replenishment elapsed lead time", "Units": f"{period_str}s"},
        {"Symbol": "$\\sigma_L$", "Name": "Lead Time Std Dev", "Definition": "Standard deviation of replenishment lead time duration", "Units": f"{period_str}s"},
        {"Symbol": "$\\mu_{DLT}$", "Name": "Mean Lead Time Demand", "Definition": "Expected total demand incurred during replenishment lead time: d * L", "Units": "units"},
        {"Symbol": "$\\sigma_{DLT}$", "Name": "Lead Time Demand Std Dev", "Definition": "Combined uncertainty: sqrt(L * sigma_d^2 + d^2 * sigma_L^2)", "Units": "units"},
        {"Symbol": "$z$", "Name": "Safety Factor (Z-Score)", "Definition": "Standard normal distribution quantile for target service level: Phi^{-1}(CSL)", "Units": "standard deviations"},
        {"Symbol": "$\\text{SS}$", "Name": "Safety Stock", "Definition": "Buffer stock held to protect against lead time stockouts: z * sigma_{DLT}", "Units": "units"},
        {"Symbol": "$\\text{ROP}$", "Name": "Reorder Point", "Definition": "Inventory level threshold triggering replenishment order placement: mu_{DLT} + SS", "Units": "units"},
        {"Symbol": "$E[US]$", "Name": "Expected Short per Cycle", "Definition": "Expected unmet units when lead time demand exceeds ROP: sigma_{DLT} * L(z)", "Units": "units/cycle"},
    ]

    base_formula = (
        "- **Mean Lead Time Demand:** $\\mu_{DLT} = \\bar{d} \\times \\bar{L}$\n"
        "- **Lead Time Demand Std Dev:** $\\sigma_{DLT} = \\sqrt{\\bar{L} \\sigma_d^2 + \\bar{d}^2 \\sigma_L^2}$ (or $\\sigma_d \\sqrt{\\bar{L}}$ when lead time is constant)\n"
        "- **Target Z-Score:** $z = \\Phi^{-1}(\\text{CSL})$\n"
        "- **Safety Stock Requirement:** $SS = z \\times \\sigma_{DLT}$\n"
        "- **Reorder Point (ROP):** $\\text{ROP} = \\mu_{DLT} + SS$\n"
        "- **Expected Units Short per Cycle:** $E[US] = \\sigma_{DLT} \\times L(z)$"
    )

    if sig_L_val > 0:
        sig_dlt_str = f"\\sqrt{{{L_val:.1f} \\times ({sig_d:.2f})^2 + ({d_val:.1f})^2 \\times ({sig_L_val:.2f})^2}} = \\sqrt{{{L_val * (sig_d**2):.2f} + {(d_val**2) * (sig_L_val**2):.2f}}} = {sigma_LT:.2f}"
    else:
        sig_dlt_str = f"{sig_d:.2f} \\times \\sqrt{{{L_val:.1f}}} = {sigma_LT:.2f}"

    substituted_formula = (
        f"- **Mean Lead Time Demand:** $\\mu_{{DLT}} = {d_val:.1f} \\times {L_val:.1f} = {lead_time_demand_mean:,.1f}$ units\n"
        f"- **Lead Time Demand Std Dev:** $\\sigma_{{DLT}} = {sig_dlt_str}$ units\n"
        f"- **Target Z-Score:** $z = \\Phi^{{-1}}({calc_csl:.2f}) = {calc_z:.4f}$\n"
        f"- **Safety Stock:** $SS = {calc_z:.4f} \\times {sigma_LT:.2f} = {ss:,.1f}$ units\n"
        f"- **Reorder Point:** $\\text{{ROP}} = {lead_time_demand_mean:,.1f} + {ss:,.1f} = {rop:,.1f}$ units"
    )

    intermediate_steps = [
        {"step": 1, "name": "Mean Lead Time Demand (mu_DLT)", "formula": "d_bar * L_bar", "value": lead_time_demand_mean, "formatted_value": f"{lead_time_demand_mean:,.2f} units", "description": "Expected customer demand consumed during replenishment lead time"},
        {"step": 2, "name": "Demand Variance Component", "formula": "L_bar * sigma_d^2", "value": L_val * (sig_d ** 2), "formatted_value": f"{L_val * (sig_d ** 2):,.4f}", "description": "Variance contribution arising from period demand volatility"},
        {"step": 3, "name": "Lead Time Variance Component", "formula": "d_bar^2 * sigma_L^2", "value": (d_val ** 2) * (sig_L_val ** 2), "formatted_value": f"{(d_val ** 2) * (sig_L_val ** 2):,.4f}", "description": "Variance contribution arising from lead time duration variability"},
        {"step": 4, "name": "Total Lead Time Std Dev (sigma_DLT)", "formula": "sqrt(Var_DLT)", "value": sigma_LT, "formatted_value": f"{sigma_LT:,.4f} units", "description": "Combined standard deviation of lead time demand"},
        {"step": 5, "name": "Safety Factor Z-Score", "formula": "Phi^{-1}(CSL)", "value": calc_z, "formatted_value": f"{calc_z:.4f}", "description": f"Standard normal quantile corresponding to {calc_csl*100:.1f}% service level"},
        {"step": 6, "name": "Safety Stock Requirement (SS)", "formula": "z * sigma_DLT", "value": ss, "formatted_value": f"{ss:,.2f} units", "description": "Buffer inventory required to guarantee target in-stock cycle probability"},
        {"step": 7, "name": "Reorder Point (ROP)", "formula": "mu_DLT + SS", "value": rop, "formatted_value": f"{rop:,.2f} units", "description": "Action threshold: trigger purchase order when inventory position reaches ROP"},
        {"step": 8, "name": "Unit Normal Loss L(z)", "formula": "phi(z) - z * (1 - Phi(z))", "value": loss_val, "formatted_value": f"{loss_val:.5f}", "description": "Normalized expected units short for standard normal distribution"},
        {"step": 9, "name": "Expected Units Short per Cycle", "formula": "sigma_DLT * L(z)", "value": expected_short, "formatted_value": f"{expected_short:.4f} units", "description": "Expected unfulfilled demand per replenishment order cycle"},
    ]

    final_result = {
        "reorder_point": rop,
        "safety_stock": ss,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_std": sigma_LT,
        "z_score": calc_z,
        "service_level": calc_csl,
        "stockout_risk_per_cycle": stockout_risk,
        "expected_short_per_cycle": expected_short,
        "case_type": case_name,
        "verification_invariants": verification,
        "is_valid": verification.get("is_valid", True),
    }

    final_summary = [
        f"- **Reorder Point (ROP):** `{rop:,.1f} units` (Trigger replenishment when inventory drops to this level)",
        f"- **Safety Stock Required (SS):** `{ss:,.1f} units`",
        f"- **Mean Lead Time Demand:** `{lead_time_demand_mean:,.1f} units` (Lead Time: `{L_val:.1f} {period_str}s`)",
        f"- **Lead Time Demand Uncertainty:** `\\sigma_{{DLT}} = {sigma_LT:,.2f} units`",
        f"- **Cycle Service Level:** `{calc_csl*100:.1f}%` (Stockout risk per replenishment cycle: `{stockout_risk*100:.1f}%`)",
        f"- **Expected Short per Cycle:** `{expected_short:.3f} units`",
    ]

    audit_md = _format_audit_markdown(
        model_name="Safety Stock and Reorder Point (Continuous Review)",
        inputs_list=inputs_list,
        var_defs=var_defs,
        base_formula=base_formula,
        substituted_formula=substituted_formula,
        intermediate_steps=intermediate_steps,
        final_result_summary=final_summary,
        verification=verification,
    )

    return AnalystVerificationAudit(
        model_name="Safety Stock and Reorder Point (Continuous Review)",
        extracted_inputs_and_units=inputs_list,
        variable_definitions=var_defs,
        base_formula=base_formula,
        formula_with_substituted_values=substituted_formula,
        intermediate_calculations=intermediate_steps,
        final_verified_result=final_result,
        audit_text=audit_md,
    )


def _generate_periodic_review_audit(
    R_val: float,
    L_val: float,
    protection_period: float,
    d_val: float,
    sig_d: float,
    demand_during_protection_mean: float,
    demand_during_protection_std: float,
    calc_z: float,
    calc_csl: float,
    ss: float,
    target_inventory: float,
    on_hand: float,
    on_order: float,
    backorders: float,
    inv_pos: float,
    order_quantity: float,
    avg_cycle_stock: float,
    avg_total_inventory: float,
    ss_ratio: float,
    period_str: str,
    verification: Dict[str, Any],
    demand_conversion_note: Optional[str] = None,
) -> AnalystVerificationAudit:
    d_val_str = f"{d_val:,.1f}" + (f" ({demand_conversion_note})" if demand_conversion_note else "")
    inputs_list = [
        {"Parameter": "Review Period", "Symbol": "$R$", "Value": f"{R_val:.0f}", "Units": f"{period_str}s"},
        {"Parameter": "Supplier Lead Time", "Symbol": "$L$", "Value": f"{L_val:.0f}", "Units": f"{period_str}s"},
        {"Parameter": "Average Demand per Period", "Symbol": "$\\bar{d}$", "Value": d_val_str, "Units": f"units/{period_str}"},
        {"Parameter": "Demand Std Dev per Period", "Symbol": "$\\sigma_d$", "Value": f"{sig_d:,.2f}", "Units": f"units/{period_str}"},
        {"Parameter": "Target Service Level", "Symbol": "$\\text{CSL}$", "Value": f"{calc_csl*100:.1f}%", "Units": "% in-stock probability"},
        {"Parameter": "Current On-Hand Stock", "Symbol": "$\\text{OH}$", "Value": f"{on_hand:,.1f}", "Units": "units"},
        {"Parameter": "Current On-Order Stock", "Symbol": "$\\text{OO}$", "Value": f"{on_order:,.1f}", "Units": "units"},
        {"Parameter": "Current Backorders", "Symbol": "$\\text{BO}$", "Value": f"{backorders:,.1f}", "Units": "units"},
        {"Parameter": "Current Inventory Position", "Symbol": "$\\text{IP}$", "Value": f"{inv_pos:,.1f}", "Units": "units"},
    ]

    var_defs = [
        {"Symbol": "$R$", "Name": "Review Period Duration", "Definition": "Elapsed time interval between successive scheduled inventory reviews", "Units": f"{period_str}s"},
        {"Symbol": "$L$", "Name": "Replenishment Lead Time", "Definition": "Elapsed time from order placement at review epoch until arrival", "Units": f"{period_str}s"},
        {"Symbol": "$T_p$", "Name": "Protection Interval", "Definition": "Total vulnerability window exposed to stockout: R + L", "Units": f"{period_str}s"},
        {"Symbol": "$\\bar{d}$", "Name": "Mean Demand per Period", "Definition": "Average customer demand rate per planning period", "Units": f"units/{period_str}"},
        {"Symbol": "$\\sigma_d$", "Name": "Demand Std Dev per Period", "Definition": "Standard deviation of demand per period", "Units": f"units/{period_str}"},
        {"Symbol": "$\\mu_p$", "Name": "Protection Demand Mean", "Definition": "Expected customer demand consumed across protection interval: d * (R + L)", "Units": "units"},
        {"Symbol": "$\\sigma_p$", "Name": "Protection Demand Std Dev", "Definition": "Combined demand volatility across protection window: sigma_d * sqrt(R + L)", "Units": "units"},
        {"Symbol": "$z$", "Name": "Safety Factor (Z-Score)", "Definition": "Standard normal distribution quantile for target service level: Phi^{-1}(CSL)", "Units": "standard deviations"},
        {"Symbol": "$SS$", "Name": "Safety Stock Requirement", "Definition": "Buffer stock held across review plus lead time: z * sigma_p", "Units": "units"},
        {"Symbol": "$S$", "Name": "Order-Up-To Level (Target)", "Definition": "Target ceiling inventory level to which IP is restored: mu_p + SS", "Units": "units"},
        {"Symbol": "$\\text{IP}$", "Name": "Inventory Position", "Definition": "Net inventory commitment: On-Hand + On-Order - Backorders", "Units": "units"},
        {"Symbol": "$q$", "Name": "Order Quantity", "Definition": "Replenishment batch placed at review epoch: max(0, S - IP)", "Units": "units"},
    ]

    base_formula = (
        "- **Protection Interval Duration:** $T_p = R + L$\n"
        "- **Mean Demand during Protection Period:** $\\mu_p = \\bar{d} (R + L)$\n"
        "- **Demand Std Dev during Protection Period:** $\\sigma_p = \\sigma_d \\sqrt{R + L}$\n"
        "- **Safety Stock Requirement:** $SS = z \\times \\sigma_p$\n"
        "- **Order-Up-To Level (Target Inventory):** $S = \\mu_p + SS$\n"
        "- **Current Inventory Position:** $\\text{IP} = \\text{On-Hand} + \\text{On-Order} - \\text{Backorders}$\n"
        "- **Recommended Replenishment Order Quantity:** $q = \\max(0, S - \\text{IP})$"
    )

    substituted_formula = (
        f"- **Protection Interval Duration:** $T_p = {R_val:.0f} + {L_val:.0f} = {protection_period:.0f}$ {period_str}s\n"
        f"- **Mean Demand during Protection Period:** $\\mu_p = {d_val:.1f} \\times {protection_period:.0f} = {demand_during_protection_mean:,.1f}$ units\n"
        f"- **Demand Std Dev during Protection Period:** $\\sigma_p = {sig_d:.1f} \\times \\sqrt{{{protection_period:.0f}}} = {demand_during_protection_std:,.2f}$ units\n"
        f"- **Safety Stock:** $SS = {calc_z:.4f} \\times {demand_during_protection_std:,.2f} = {ss:,.1f}$ units\n"
        f"- **Order-Up-To Level (S):** $S = {demand_during_protection_mean:,.1f} + {ss:,.1f} = {target_inventory:,.1f}$ units\n"
        f"- **Current Inventory Position:** $\\text{{IP}} = {on_hand:,.1f} + {on_order:,.1f} - {backorders:,.1f} = {inv_pos:,.1f}$ units\n"
        f"- **Recommended Order Quantity:** $q = \\max(0, {target_inventory:,.1f} - {inv_pos:,.1f}) = {order_quantity:,.1f}$ units"
    )

    intermediate_steps = [
        {"step": 1, "name": "Protection Interval (R + L)", "formula": "R + L", "value": protection_period, "formatted_value": f"{protection_period:.0f} {period_str}s", "description": "Total duration over which inventory position must protect against stockout"},
        {"step": 2, "name": "Mean Protection Demand (mu_p)", "formula": "d_bar * (R + L)", "value": demand_during_protection_mean, "formatted_value": f"{demand_during_protection_mean:,.2f} units", "description": "Expected customer consumption across review interval plus lead time"},
        {"step": 3, "name": "Protection Demand Std Dev (sigma_p)", "formula": "sigma_d * sqrt(R + L)", "value": demand_during_protection_std, "formatted_value": f"{demand_during_protection_std:,.4f} units", "description": "Aggregated standard deviation across the protection interval"},
        {"step": 4, "name": "Safety Factor Z-Score", "formula": "Phi^{-1}(CSL)", "value": calc_z, "formatted_value": f"{calc_z:.4f}", "description": f"Standard normal quantile for {calc_csl*100:.1f}% service level"},
        {"step": 5, "name": "Safety Stock Requirement (SS)", "formula": "z * sigma_p", "value": ss, "formatted_value": f"{ss:,.2f} units", "description": "Buffer stock required across review plus lead time window"},
        {"step": 6, "name": "Order-Up-To Level (S)", "formula": "mu_p + SS", "value": target_inventory, "formatted_value": f"{target_inventory:,.2f} units", "description": "Target replenishment baseline restoring inventory position at review"},
        {"step": 7, "name": "Inventory Position (IP)", "formula": "OH + OO - BO", "value": inv_pos, "formatted_value": f"{inv_pos:,.2f} units", "description": "Effective available inventory commitment at review epoch"},
        {"step": 8, "name": "Unconstrained Order Deficit", "formula": "S - IP", "value": target_inventory - inv_pos, "formatted_value": f"{target_inventory - inv_pos:,.2f} units", "description": "Net difference between target inventory ceiling and current inventory position"},
        {"step": 9, "name": "Clipped Recommended Order (q)", "formula": "max(0, S - IP)", "value": order_quantity, "formatted_value": f"{order_quantity:,.2f} units", "description": "Actual batch size placed with supplier (non-negative constraint enforced)"},
        {"step": 10, "name": "Periodic vs Continuous SS Ratio", "formula": "SS_periodic / SS_continuous", "value": ss_ratio, "formatted_value": f"{ss_ratio:.4f}x", "description": "Relative safety stock premium necessitated by periodic review delays"},
    ]

    final_result = {
        "recommended_order_quantity": order_quantity,
        "target_inventory": target_inventory,
        "order_up_to_level": target_inventory,
        "safety_stock": ss,
        "protection_period": protection_period,
        "inventory_position": inv_pos,
        "average_cycle_stock": avg_cycle_stock,
        "average_total_inventory": avg_total_inventory,
        "comparison_vs_continuous_review_ss": ss_ratio,
        "verification_invariants": verification,
        "is_valid": verification.get("is_valid", True),
    }

    final_summary = [
        f"- **Recommended Order Quantity ($q$):** `{order_quantity:,.1f} units` (Order placed this review cycle)",
        f"- **Order-Up-To Level ($S$ / Target Inventory):** `{target_inventory:,.1f} units`",
        f"- **Safety Stock Buffer ($SS$):** `{ss:,.1f} units`",
        f"- **Protection Period:** `{protection_period:.0f} {period_str}s` (Review: `{R_val:.0f}` + Lead Time: `{L_val:.0f}`)",
        f"- **Current Inventory Position:** `{inv_pos:,.1f} units` (On-Hand: `{on_hand:,.1f}`, On-Order: `{on_order:,.1f}`, Backorders: `{backorders:,.1f}`)",
        f"- **Average Operational Inventory:** Cycle Stock: `{avg_cycle_stock:,.1f} units` | Total Stock: `{avg_total_inventory:,.1f} units`",
    ]

    audit_md = _format_audit_markdown(
        model_name="Fixed-Period Review Model (Periodic Review / Order-Up-To)",
        inputs_list=inputs_list,
        var_defs=var_defs,
        base_formula=base_formula,
        substituted_formula=substituted_formula,
        intermediate_steps=intermediate_steps,
        final_result_summary=final_summary,
        verification=verification,
    )

    return AnalystVerificationAudit(
        model_name="Fixed-Period Review Model (Periodic Review / Order-Up-To)",
        extracted_inputs_and_units=inputs_list,
        variable_definitions=var_defs,
        base_formula=base_formula,
        formula_with_substituted_values=substituted_formula,
        intermediate_calculations=intermediate_steps,
        final_verified_result=final_result,
        audit_text=audit_md,
    )


def _generate_newsvendor_audit(
    p_val: float,
    c_val: float,
    v_val: float,
    g_val: float,
    cu: float,
    co: float,
    calc_cr: float,
    mu_val: float,
    sig_val: float,
    z_star: float,
    q_star: float,
    loss_z: float,
    exp_lost_sales: float,
    exp_sales: float,
    exp_leftover: float,
    exp_rev: float,
    exp_procurement: float,
    exp_salvage: float,
    exp_penalty: float,
    exp_profit: float,
    max_profit: float,
    evpi: float,
    verification: Dict[str, Any],
) -> AnalystVerificationAudit:
    inputs_list = [
        {"Parameter": "Unit Selling Price", "Symbol": "$P$", "Value": f"\\${p_val:,.2f}", "Units": "$/unit"},
        {"Parameter": "Unit Purchase Cost", "Symbol": "$C$", "Value": f"\\${c_val:,.2f}", "Units": "$/unit"},
        {"Parameter": "Unit Salvage Value", "Symbol": "$V$", "Value": f"\\${v_val:,.2f}", "Units": "$/unit"},
        {"Parameter": "Unit Stockout Penalty", "Symbol": "$g$", "Value": f"\\${g_val:,.2f}", "Units": "$/unit"},
        {"Parameter": "Mean Period Demand", "Symbol": "$\\mu$", "Value": f"{mu_val:,.1f}", "Units": "units"},
        {"Parameter": "Demand Std Dev", "Symbol": "$\\sigma$", "Value": f"{sig_val:,.1f}", "Units": "units"},
    ]

    var_defs = [
        {"Symbol": "$P$", "Name": "Unit Selling Price", "Definition": "Revenue collected per unit sold during selling season", "Units": "$/unit"},
        {"Symbol": "$C$", "Name": "Unit Purchase / Production Cost", "Definition": "Direct acquisition expense incurred per unit ordered", "Units": "$/unit"},
        {"Symbol": "$V$", "Name": "Unit Salvage Value", "Definition": "Net liquidation price recovered per unsold leftover unit", "Units": "$/unit"},
        {"Symbol": "$g$", "Name": "Stockout Penalty", "Definition": "Goodwill loss or penalty cost per unit of unsatisfied customer demand", "Units": "$/unit"},
        {"Symbol": "$C_u$", "Name": "Cost of Underage (Lost Margin)", "Definition": "Marginal profit opportunity lost per unit short: P - C + g", "Units": "$/unit"},
        {"Symbol": "$C_o$", "Name": "Cost of Overage (Disposal Loss)", "Definition": "Marginal loss incurred on each unsold excess unit: C - V", "Units": "$/unit"},
        {"Symbol": "$\\text{CR}$", "Name": "Critical Ratio (Critical Fractile)", "Definition": "Optimal in-stock cycle probability threshold: C_u / (C_u + C_o)", "Units": "probability (0 to 1)"},
        {"Symbol": "$\\mu$", "Name": "Mean Period Demand", "Definition": "Expected customer demand in single selling season", "Units": "units"},
        {"Symbol": "$\\sigma$", "Name": "Demand Std Dev", "Definition": "Demand uncertainty standard deviation in single selling season", "Units": "units"},
        {"Symbol": "$z^*$", "Name": "Optimal Safety Factor", "Definition": "Standard normal distribution quantile for critical ratio: Phi^{-1}(CR)", "Units": "standard deviations"},
        {"Symbol": "$Q^*$", "Name": "Optimal Order Quantity", "Definition": "Profit-maximizing procurement lot size: mu + z^* * sigma", "Units": "units"},
        {"Symbol": "$E[\\text{Sales}]$", "Name": "Expected Sales", "Definition": "Expected units sold to customers: mu - sigma * L(z^*)", "Units": "units"},
        {"Symbol": "$E[\\text{Leftover}]$", "Name": "Expected Waste (Leftover)", "Definition": "Expected unsold units liquidated at salvage: Q^* - E[Sales]", "Units": "units"},
        {"Symbol": "$E[\\text{Profit}]$", "Name": "Expected Total Profit", "Definition": "Expected revenue + salvage recovery - procurement - penalties", "Units": "$"},
        {"Symbol": "$\\text{EVPI}$", "Name": "Value of Perfect Information", "Definition": "Maximum additional profit capturable if demand were known with certainty", "Units": "$"},
    ]

    base_formula = (
        "- **Cost of Underage:** $C_u = P - C + g$\n"
        "- **Cost of Overage:** $C_o = C - V$\n"
        "- **Critical Ratio (Critical Fractile):** $\\text{CR} = \\frac{C_u}{C_u + C_o}$\n"
        "- **Optimal Standard Normal Safety Factor:** $z^* = \\Phi^{-1}(\\text{CR})$\n"
        "- **Optimal Procurement Lot Size:** $Q^* = \\mu + z^* \\sigma$\n"
        "- **Expected Sales Volume:** $E[\\text{Sales}] = \\mu - \\sigma L(z^*)$\n"
        "- **Expected Leftover Inventory:** $E[\\text{Leftover}] = Q^* - E[\\text{Sales}]$\n"
        "- **Expected Financial Profit:** $E[\\text{Profit}] = P \\cdot E[\\text{Sales}] + V \\cdot E[\\text{Leftover}] - C \\cdot Q^* - g \\cdot E[\\text{Lost Sales}]$\n"
        "- **Expected Value of Perfect Information:** $\\text{EVPI} = (P - C) \\mu - E[\\text{Profit}]$"
    )

    substituted_formula = (
        f"- **Cost of Underage:** $C_u = P - C + g =$ \\${p_val:,.2f} - \\${c_val:,.2f} + \\${g_val:,.2f} = \\${cu:,.2f}\n"
        f"- **Cost of Overage:** $C_o = C - V =$ \\${c_val:,.2f} - \\${v_val:,.2f} = \\${co:,.2f}\n"
        f"- **Critical Fractile:** $\\text{{CR}} = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{cu:,.2f}}}{{{cu + co:,.2f}}} = {calc_cr:.4f}$ ({calc_cr*100:.2f}%)\n"
        f"- **Optimal Z-Score:** $z^* = \\Phi^{{-1}}({calc_cr:.4f}) = {z_star:.4f}$\n"
        f"- **Optimal Order Quantity:** $Q^* = \\mu + z^* \\sigma = {mu_val:,.1f} + ({z_star:.4f} \\times {sig_val:,.1f}) = {q_star:,.1f}$ units\n"
        f"- **Expected Financial Profit:** $E[\\text{{Profit}}] =$ \\${exp_profit:,.2f}\n"
        f"- **Value of Perfect Information:** $\\text{{EVPI}} =$ \\${evpi:,.2f}"
    )

    intermediate_steps = [
        {"step": 1, "name": "Marginal Underage Cost (Cu)", "formula": "P - C + g", "value": cu, "formatted_value": f"${cu:,.2f}", "description": "Profit contribution lost per unit of unmet customer demand"},
        {"step": 2, "name": "Marginal Overage Cost (Co)", "formula": "C - V", "value": co, "formatted_value": f"${co:,.2f}", "description": "Net financial loss absorbed per unit of unsold inventory"},
        {"step": 3, "name": "Total Marginal Cost Pool (Cu + Co)", "formula": "Cu + Co", "value": cu + co, "formatted_value": f"${cu + co:,.2f}", "description": "Denominator balancing marginal underage and overage penalties"},
        {"step": 4, "name": "Critical Ratio Fractile (CR)", "formula": "Cu / (Cu + Co)", "value": calc_cr, "formatted_value": f"{calc_cr:.4f} ({calc_cr*100:.2f}%)", "description": "Optimal non-stockout service probability balancing marginal trade-offs"},
        {"step": 5, "name": "Optimal Safety Factor (z*)", "formula": "Phi^{-1}(CR)", "value": z_star, "formatted_value": f"{z_star:.4f}", "description": "Standard normal distribution quantile for critical fractile"},
        {"step": 6, "name": "Safety Buffer Allowance (z* * sigma)", "formula": "z* * sigma", "value": z_star * sig_val, "formatted_value": f"{z_star * sig_val:,.2f} units", "description": "Buffer units added to (or subtracted from) mean demand"},
        {"step": 7, "name": "Optimal Order Quantity (Q*)", "formula": "mu + z* * sigma", "value": q_star, "formatted_value": f"{q_star:,.2f} units", "description": "Expected profit-maximizing order quantity"},
        {"step": 8, "name": "Standard Normal Unit Loss L(z*)", "formula": "phi(z*) - z* * (1 - Phi(z*))", "value": loss_z, "formatted_value": f"{loss_z:.5f}", "description": "Expected unit loss factor for standard normal distribution"},
        {"step": 9, "name": "Expected Lost Sales", "formula": "sigma * L(z*)", "value": exp_lost_sales, "formatted_value": f"{exp_lost_sales:,.2f} units", "description": "Expected unsatisfied customer demand units"},
        {"step": 10, "name": "Expected Sales Volume", "formula": "mu - Expected Lost Sales", "value": exp_sales, "formatted_value": f"{exp_sales:,.2f} units", "description": "Expected units successfully sold at full retail price"},
        {"step": 11, "name": "Expected Leftover Inventory", "formula": "Q* - Expected Sales", "value": exp_leftover, "formatted_value": f"{exp_leftover:,.2f} units", "description": "Expected unsold units requiring salvage disposal"},
        {"step": 12, "name": "Inventory Balance Invariant Check", "formula": "Sales + Leftover == Q*", "value": exp_sales + exp_leftover, "formatted_value": f"{exp_sales + exp_leftover:,.2f} units", "description": "Conservation invariant: sales plus leftovers must equal total batch size"},
        {"step": 13, "name": "Expected Sales Revenue", "formula": "P * E[Sales]", "value": exp_rev, "formatted_value": f"${exp_rev:,.2f}", "description": "Gross cash receipts generated from full-price units"},
        {"step": 14, "name": "Expected Procurement Cost", "formula": "C * Q*", "value": exp_procurement, "formatted_value": f"${exp_procurement:,.2f}", "description": "Total cash paid upfront to supplier for procurement batch"},
        {"step": 15, "name": "Expected Salvage Recovery", "formula": "V * E[Leftover]", "value": exp_salvage, "formatted_value": f"${exp_salvage:,.2f}", "description": "Cash recovered through liquidation of excess units"},
        {"step": 16, "name": "Expected Financial Profit", "formula": "Revenue + Salvage - Procurement - Penalties", "value": exp_profit, "formatted_value": f"${exp_profit:,.2f}", "description": "Net expected operating profit from single-period sales"},
        {"step": 17, "name": "Expected Value of Perfect Information (EVPI)", "formula": "(P - C) * mu - E[Profit]", "value": evpi, "formatted_value": f"${evpi:,.2f}", "description": "Theoretical maximum value of eliminating demand forecast uncertainty"},
    ]

    final_result = {
        "recommended_order_quantity": q_star,
        "optimal_order_quantity": q_star,
        "critical_ratio": calc_cr,
        "optimal_z_score": z_star,
        "cost_of_underage": cu,
        "cost_of_overage": co,
        "expected_sales": exp_sales,
        "expected_lost_sales": exp_lost_sales,
        "expected_leftover_units": exp_leftover,
        "expected_revenue": exp_rev,
        "expected_procurement_cost": exp_procurement,
        "expected_salvage_value": exp_salvage,
        "expected_profit": exp_profit,
        "maximum_possible_profit": max_profit,
        "expected_value_of_perfect_information": evpi,
        "verification_invariants": verification,
        "is_valid": verification.get("is_valid", True),
    }

    final_summary = [
        f"- **Recommended Order Quantity ($Q^*$):** `{q_star:,.1f} units`",
        f"- **Critical Ratio (In-Stock Probability):** `{calc_cr*100:.2f}%` ($z^* = {z_star:.4f}$)",
        f"- **Marginal Economics:** Cost of Underage ($C_u$): `\\${cu:,.2f}` | Cost of Overage ($C_o$): `\\${co:,.2f}`",
        f"- **Expected Operational Outcomes:** Sales: `{exp_sales:,.1f} units` | Leftover Waste: `{exp_leftover:,.1f} units`",
        f"- **Expected Financial Profit:** `\\${exp_profit:,.2f}` (Perfect Info Benchmark: `\\${max_profit:,.2f}`)",
        f"- **Expected Value of Perfect Information (EVPI):** `\\${evpi:,.2f}`",
    ]

    audit_md = _format_audit_markdown(
        model_name="Newsvendor Model (Single-Period Stochastic)",
        inputs_list=inputs_list,
        var_defs=var_defs,
        base_formula=base_formula,
        substituted_formula=substituted_formula,
        intermediate_steps=intermediate_steps,
        final_result_summary=final_summary,
        verification=verification,
    )

    return AnalystVerificationAudit(
        model_name="Newsvendor Model (Single-Period Stochastic)",
        extracted_inputs_and_units=inputs_list,
        variable_definitions=var_defs,
        base_formula=base_formula,
        formula_with_substituted_values=substituted_formula,
        intermediate_calculations=intermediate_steps,
        final_verified_result=final_result,
        audit_text=audit_md,
    )


# ============================================================================
# 1. Economic Order Quantity (EOQ)
# ============================================================================

@dataclass
class EOQInputs:
    annual_demand: float
    order_cost: float
    holding_cost: float
    unit_cost: float = 0.0
    lead_time_days: float = 0.0
    work_days_per_year: int = 250
    demand_conversion_note: Optional[str] = None
    holding_conversion_note: Optional[str] = None


@dataclass
class EOQResult:
    final_answer: float                  # Q* (optimal order quantity)
    eoq: float                           # alias for final_answer
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    inputs: Dict[str, Any]
    orders_per_year: float
    order_interval_years: float
    order_interval_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_inventory_cost: float
    annual_purchase_cost: float
    total_annual_cost: float
    reorder_point: float
    verification: Dict[str, Any]
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    annual_demand: Optional[Union[float, EOQInputs]] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: Optional[int] = None,
    # Support D, S, H, C aliases
    D: Optional[float] = None,
    S: Optional[float] = None,
    H: Optional[float] = None,
    C: Optional[float] = None,
    demand_conversion_note: Optional[str] = None,
    holding_conversion_note: Optional[str] = None,
) -> EOQResult:
    """
    Calculates the Economic Order Quantity (EOQ):
    Q* = sqrt(2 * D * S / H)

    Validates inputs, returns intermediate calculations, final answer,
    and identifies all operational assumptions. Never invents missing numerical inputs.
    """
    # Allow EOQInputs dataclass as first positional parameter
    if isinstance(annual_demand, EOQInputs):
        in_obj = annual_demand
        demand = in_obj.annual_demand
        setup_cost = in_obj.order_cost
        hold_cost = in_obj.holding_cost
        item_cost = in_obj.unit_cost
        lead_time = in_obj.lead_time_days
        work_days = in_obj.work_days_per_year
        demand_conv = in_obj.demand_conversion_note or demand_conversion_note
        hold_conv = in_obj.holding_conversion_note or holding_conversion_note
    else:
        demand = annual_demand if annual_demand is not None else D
        setup_cost = order_cost if order_cost is not None else S
        hold_cost = holding_cost if holding_cost is not None else H
        item_cost = unit_cost if unit_cost is not None else C
        lead_time = lead_time_days
        work_days = work_days_per_year
        demand_conv = demand_conversion_note
        hold_conv = holding_conversion_note

    # Strict input validation - never invent missing numerical inputs
    if demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' (or 'D') must be provided.")
    if setup_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (or 'S') must be provided.")
    if hold_cost is None:
        raise ValueError("Missing required numerical input: 'holding_cost' (or 'H') must be provided.")

    try:
        demand = float(demand)
        setup_cost = float(setup_cost)
        hold_cost = float(hold_cost)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Inputs must be valid numerical values. Conversion error: {e}")

    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0). Got {demand}")
    if setup_cost <= 0:
        raise ValueError(f"Order cost must be strictly positive (> 0). Got {setup_cost}")
    if hold_cost <= 0:
        raise ValueError(f"Holding cost must be strictly positive (> 0). Got {hold_cost}")

    item_cost_val = float(item_cost) if item_cost is not None else 0.0
    if item_cost_val < 0:
        raise ValueError(f"Unit cost cannot be negative. Got {item_cost_val}")

    lead_time_val = float(lead_time) if lead_time is not None else 0.0
    if lead_time_val < 0:
        raise ValueError(f"Lead time cannot be negative. Got {lead_time_val}")

    work_days_val = int(work_days) if work_days is not None else 250
    if work_days_val <= 0:
        raise ValueError(f"Work days per year must be strictly positive. Got {work_days_val}")

    # Core EOQ calculation: Q = sqrt(2DS/H)
    numerator = 2.0 * demand * setup_cost
    q_star = math.sqrt(numerator / hold_cost)

    # Intermediate calculations
    orders_per_year = demand / q_star
    order_interval_years = q_star / demand
    order_interval_days = order_interval_years * work_days_val
    annual_ordering_cost = orders_per_year * setup_cost
    annual_holding_cost = (q_star / 2.0) * hold_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = demand * item_cost_val
    total_cost = total_inventory_cost + annual_purchase_cost

    daily_demand = demand / work_days_val
    rop = daily_demand * lead_time_val

    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    cost_diff_rel = cost_diff / annual_holding_cost if annual_holding_cost > 0 else 0.0
    second_derivative = (2.0 * demand * setup_cost) / (q_star ** 3)

    def tc_at(q: float) -> float:
        return (demand / q) * setup_cost + (q / 2.0) * hold_cost

    sensitivity = {
        "minus_20pct_qty": tc_at(q_star * 0.8),
        "minus_10pct_qty": tc_at(q_star * 0.9),
        "optimal_qty": total_inventory_cost,
        "plus_10pct_qty": tc_at(q_star * 1.1),
        "plus_20pct_qty": tc_at(q_star * 1.2),
        "penalty_minus_10pct": (tc_at(q_star * 0.9) - total_inventory_cost) / total_inventory_cost,
        "penalty_plus_10pct": (tc_at(q_star * 1.1) - total_inventory_cost) / total_inventory_cost,
    }

    intermediates = {
        "formula": "Q = sqrt(2 * D * S / H)",
        "numerator_2DS": numerator,
        "holding_cost_H": hold_cost,
        "orders_per_year": orders_per_year,
        "order_interval_years": order_interval_years,
        "order_interval_days": order_interval_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_annual_inventory_cost": total_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_cost,
        "daily_demand": daily_demand,
        "reorder_point": rop,
        "ordering_equals_holding_diff": cost_diff,
        "second_derivative_d2TC_dQ2": second_derivative,
        "is_global_convex_minimum": second_derivative > 0,
        "sensitivity": sensitivity,
    }

    assumptions = [
        "Demand rate is known, constant, deterministic, and continuous over the entire annual horizon.",
        "Ordering / setup cost is a fixed dollar amount per replenishment order regardless of quantity ordered.",
        "Inventory carrying / holding cost is constant per unit per year.",
        "Entire replenishment lot Q arrives instantaneously in a single batch (instantaneous replenishment).",
        "No shortages, backorders, or stockouts are permitted.",
        "No quantity discounts apply (unit purchase price is flat).",
    ]

    inputs_recorded = {
        "annual_demand": demand,
        "order_cost": setup_cost,
        "holding_cost": hold_cost,
        "unit_cost": item_cost_val,
        "lead_time_days": lead_time_val,
        "work_days_per_year": work_days_val,
    }

    verification = {
        "is_valid": cost_diff_rel < 1e-4 and second_derivative > 0,
        "holding_equals_ordering": cost_diff_rel < 1e-4,
        "cost_difference": cost_diff,
        "relative_cost_difference": cost_diff_rel,
        "second_derivative_strictly_positive": second_derivative > 0,
        "sensitivity": sensitivity,
    }

    audit = _generate_eoq_audit(
        demand=demand,
        setup_cost=setup_cost,
        hold_cost=hold_cost,
        item_cost=item_cost_val,
        lead_time=lead_time_val,
        work_days=work_days_val,
        q_star=q_star,
        orders_per_year=orders_per_year,
        order_interval_days=order_interval_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_inventory_cost=total_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_cost=total_cost,
        rop=rop,
        verification=verification,
        demand_conversion_note=demand_conv,
        holding_conversion_note=hold_conv,
    )

    return EOQResult(
        final_answer=q_star,
        eoq=q_star,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        inputs=inputs_recorded,
        orders_per_year=orders_per_year,
        order_interval_years=order_interval_years,
        order_interval_days=order_interval_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_inventory_cost=total_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_cost,
        reorder_point=rop,
        verification=verification,
        audit=audit,
    )


# ============================================================================
# 2. Quantity Discount Model
# ============================================================================

@dataclass
class DiscountTier:
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    tier_name: str = ""


@dataclass
class TierEvaluationResult:
    tier_index: int
    tier_name: str
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_unit: float
    raw_eoq: float
    is_raw_eoq_feasible: bool
    feasible_order_quantity: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    tier: Optional[DiscountTier] = None

    @property
    def feasible_order_qty(self) -> float:
        return self.feasible_order_quantity


@dataclass
class QuantityDiscountInputs:
    annual_demand: float
    order_cost: float
    tiers: List[DiscountTier]
    holding_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    work_days_per_year: int = 250
    demand_conversion_note: Optional[str] = None
    holding_conversion_note: Optional[str] = None


@dataclass
class QuantityDiscountResult:
    final_answer: float                           # optimal order quantity Q*
    optimal_order_quantity: float                 # alias for final_answer
    optimal_tier_index: int
    optimal_unit_price: float
    optimal_total_annual_cost: float
    optimal_annual_purchase_cost: float
    optimal_annual_ordering_cost: float
    optimal_annual_holding_cost: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    inputs: Dict[str, Any]
    tier_evaluations: List[TierEvaluationResult]
    base_tier_cost: float
    annual_savings_vs_base: float
    percentage_savings_vs_base: float
    verification: Dict[str, Any]
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit

    @property
    def optimal_order_qty(self) -> float:
        return self.optimal_order_quantity

    @property
    def base_tier_total_cost(self) -> float:
        return self.base_tier_cost

    @property
    def annual_cost_savings_vs_base(self) -> float:
        return self.annual_savings_vs_base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_quantity_discount(
    annual_demand: Optional[Union[float, QuantityDiscountInputs]] = None,
    order_cost: Optional[float] = None,
    tiers: Optional[List[Union[DiscountTier, Dict[str, Any], Tuple]]] = None,
    holding_rate: Optional[float] = None,          # e.g., 0.20 for 20%
    fixed_holding_cost: Optional[float] = None,    # e.g., $2.50 per unit/year
    work_days_per_year: Optional[int] = None,
    # Support D, S aliases
    D: Optional[float] = None,
    S: Optional[float] = None,
    demand_conversion_note: Optional[str] = None,
    holding_conversion_note: Optional[str] = None,
) -> QuantityDiscountResult:
    """
    Evaluates an all-units quantity discount schedule and identifies the cost-minimizing order quantity.
    Supports both fixed holding cost and percentage holding cost.

    Validates inputs, evaluates feasible order quantities for each tier,
    compares total annual costs, and identifies all assumptions. Never invents missing numerical inputs.
    """
    if isinstance(annual_demand, QuantityDiscountInputs):
        in_obj = annual_demand
        demand = in_obj.annual_demand
        setup_cost = in_obj.order_cost
        tier_list = in_obj.tiers
        h_rate = in_obj.holding_rate
        f_hold = in_obj.fixed_holding_cost
        work_days = in_obj.work_days_per_year
        demand_conv = in_obj.demand_conversion_note or demand_conversion_note
        hold_conv = in_obj.holding_conversion_note or holding_conversion_note
    else:
        demand = annual_demand if annual_demand is not None else D
        setup_cost = order_cost if order_cost is not None else S
        tier_list = tiers
        h_rate = holding_rate
        f_hold = fixed_holding_cost
        work_days = work_days_per_year
        demand_conv = demand_conversion_note
        hold_conv = holding_conversion_note

    # Input validation
    if demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' (or 'D') must be provided.")
    if setup_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (or 'S') must be provided.")
    if tier_list is None or len(tier_list) == 0:
        raise ValueError("Missing required input: 'tiers' discount schedule must be a non-empty list.")
    if h_rate is None and f_hold is None:
        raise ValueError("Missing required input: Either 'holding_rate' or 'fixed_holding_cost' must be provided.")
    if h_rate is not None and f_hold is not None:
        raise ValueError("Ambiguous holding cost: Specify either 'holding_rate' or 'fixed_holding_cost', not both.")

    demand = float(demand)
    setup_cost = float(setup_cost)

    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive. Got {demand}")
    if setup_cost <= 0:
        raise ValueError(f"Order cost must be strictly positive. Got {setup_cost}")

    if h_rate is not None:
        h_rate = float(h_rate)
        if h_rate <= 0:
            raise ValueError(f"Holding rate must be strictly positive. Got {h_rate}")
    if f_hold is not None:
        f_hold = float(f_hold)
        if f_hold <= 0:
            raise ValueError(f"Fixed holding cost must be strictly positive. Got {f_hold}")

    # Normalize tiers into DiscountTier objects
    normalized_tiers: List[DiscountTier] = []
    for idx, t in enumerate(tier_list):
        if isinstance(t, DiscountTier):
            normalized_tiers.append(t)
        elif isinstance(t, dict):
            min_q = float(t.get("min_qty", t.get("min", 1)))
            max_q = float(t["max_qty"]) if t.get("max_qty") is not None and t.get("max_qty") != float("inf") else None
            price = float(t.get("unit_price", t.get("price", 0)))
            name = str(t.get("tier_name", f"Tier {idx+1}"))
            normalized_tiers.append(DiscountTier(min_qty=min_q, max_qty=max_q, unit_price=price, tier_name=name))
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            min_q = float(t[0])
            max_q = float(t[1]) if t[1] is not None and t[1] != float("inf") else None
            price = float(t[2])
            name = str(t[3]) if len(t) > 3 else f"Tier {idx+1}"
            normalized_tiers.append(DiscountTier(min_qty=min_q, max_qty=max_q, unit_price=price, tier_name=name))
        else:
            raise ValueError(f"Invalid tier format at index {idx}: {t}")

    tier_evaluations: List[TierEvaluationResult] = []

    for idx, tier in enumerate(normalized_tiers):
        price = float(tier.unit_price)
        if price <= 0:
            raise ValueError(f"Tier {idx} unit price must be strictly positive. Got {price}")

        if h_rate is not None:
            H_tier = h_rate * price
        else:
            H_tier = float(f_hold)  # type: ignore

        if H_tier <= 0:
            raise ValueError(f"Calculated holding cost for tier {idx} must be strictly positive. Got {H_tier}")

        raw_eoq = math.sqrt((2.0 * demand * setup_cost) / H_tier)
        min_q = float(tier.min_qty)
        max_q = float(tier.max_qty) if tier.max_qty is not None else float("inf")

        # Feasibility check for All-Units Discount
        if min_q <= raw_eoq <= max_q:
            feasible_q = raw_eoq
            is_feasible = True
        elif raw_eoq < min_q:
            feasible_q = min_q
            is_feasible = False
        else:
            feasible_q = max_q
            is_feasible = False

        orders = demand / feasible_q
        order_cost_ann = orders * setup_cost
        holding_cost_ann = (feasible_q / 2.0) * H_tier
        purchase_cost_ann = demand * price
        total_cost_ann = purchase_cost_ann + order_cost_ann + holding_cost_ann

        tier_evaluations.append(
            TierEvaluationResult(
                tier_index=idx,
                tier_name=tier.tier_name if tier.tier_name else f"Tier {idx+1}",
                min_qty=min_q,
                max_qty=tier.max_qty,
                unit_price=price,
                holding_cost_unit=H_tier,
                raw_eoq=raw_eoq,
                is_raw_eoq_feasible=is_feasible,
                feasible_order_quantity=feasible_q,
                annual_purchase_cost=purchase_cost_ann,
                annual_ordering_cost=order_cost_ann,
                annual_holding_cost=holding_cost_ann,
                total_annual_cost=total_cost_ann,
                tier=tier,
            )
        )

    # Find the cost-minimizing tier
    best_idx = 0
    min_tot_cost = float("inf")
    for idx, ev in enumerate(tier_evaluations):
        if ev.total_annual_cost < min_tot_cost:
            min_tot_cost = ev.total_annual_cost
            best_idx = idx

    optimal_tier = tier_evaluations[best_idx]
    base_cost = tier_evaluations[0].total_annual_cost
    savings = base_cost - optimal_tier.total_annual_cost
    pct_savings = (savings / base_cost) * 100.0 if base_cost > 0 else 0.0

    verification = {
        "is_valid": min_tot_cost < float("inf"),
        "best_tier_strictly_minimal": all(
            optimal_tier.total_annual_cost <= ev.total_annual_cost for ev in tier_evaluations
        ),
        "total_tiers_evaluated": len(tier_evaluations),
        "optimal_qty_in_tier_bounds": (
            optimal_tier.feasible_order_quantity >= optimal_tier.min_qty
            and (
                optimal_tier.max_qty is None
                or optimal_tier.feasible_order_quantity <= optimal_tier.max_qty
            )
        ),
    }

    intermediates = {
        "holding_cost_mechanism": "percentage_of_price" if h_rate is not None else "fixed_dollar_amount",
        "holding_rate": h_rate,
        "fixed_holding_cost": f_hold,
        "total_tiers_evaluated": len(tier_evaluations),
        "evaluated_tiers": [asdict(ev) for ev in tier_evaluations],
        "optimal_tier_index": best_idx,
        "optimal_feasible_order_quantity": optimal_tier.feasible_order_quantity,
        "optimal_annual_purchase_cost": optimal_tier.annual_purchase_cost,
        "optimal_annual_ordering_cost": optimal_tier.annual_ordering_cost,
        "optimal_annual_holding_cost": optimal_tier.annual_holding_cost,
        "optimal_total_annual_cost": optimal_tier.total_annual_cost,
        "base_tier_total_cost": base_cost,
        "annual_savings_vs_base": savings,
        "percentage_savings_vs_base": pct_savings,
    }

    assumptions = [
        "All-units quantity discount structure: price discount applies to every unit purchased in the lot once threshold is reached.",
        "Annual demand is known, deterministic, and constant across the entire planning period.",
        "Replenishment is instantaneous and full lot Q arrives together.",
        "Carrying cost is either constant across all units or proportional to unit purchase price.",
        "Shortages and backorders are not permitted.",
    ]

    inputs_recorded = {
        "annual_demand": demand,
        "order_cost": setup_cost,
        "holding_rate": h_rate,
        "fixed_holding_cost": f_hold,
        "tiers_count": len(normalized_tiers),
    }

    audit = _generate_quantity_discount_audit(
        demand=demand,
        setup_cost=setup_cost,
        h_rate=h_rate,
        f_hold=f_hold,
        tier_evaluations=tier_evaluations,
        best_idx=best_idx,
        optimal_tier=optimal_tier,
        base_cost=base_cost,
        savings=savings,
        pct_savings=pct_savings,
        verification=verification,
        demand_conversion_note=demand_conv,
        holding_conversion_note=hold_conv,
    )

    return QuantityDiscountResult(
        final_answer=optimal_tier.feasible_order_quantity,
        optimal_order_quantity=optimal_tier.feasible_order_quantity,
        optimal_tier_index=best_idx,
        optimal_unit_price=optimal_tier.unit_price,
        optimal_total_annual_cost=optimal_tier.total_annual_cost,
        optimal_annual_purchase_cost=optimal_tier.annual_purchase_cost,
        optimal_annual_ordering_cost=optimal_tier.annual_ordering_cost,
        optimal_annual_holding_cost=optimal_tier.annual_holding_cost,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        inputs=inputs_recorded,
        tier_evaluations=tier_evaluations,
        base_tier_cost=base_cost,
        annual_savings_vs_base=savings,
        percentage_savings_vs_base=pct_savings,
        verification=verification,
        audit=audit,
    )


# ============================================================================
# 3. Safety Stock and Reorder Point (ROP)
# ============================================================================

@dataclass
class SafetyStockROPInputs:
    avg_demand_per_period: float
    std_demand_per_period: float
    lead_time_periods: float
    std_lead_time_periods: float = 0.0
    service_level: float = 0.95
    period_name: str = "day"
    demand_conversion_note: Optional[str] = None
    lead_time_conversion_note: Optional[str] = None


@dataclass
class ROPResult:
    final_answer: float                 # Reorder Point (ROP)
    reorder_point: float                # alias for final_answer
    safety_stock: float
    case_type: str                      # one of 4 cases
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    inputs: Dict[str, Any]
    lead_time_demand_mean: float
    lead_time_demand_std: float
    z_score: float
    service_level: Optional[float]
    stockout_risk_per_cycle: float
    expected_short_per_cycle: float
    normal_loss_value: float
    period_name: str
    verification: Dict[str, Any]
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backwards compatibility type alias
SafetyStockROPResult = ROPResult


def calculate_rop(
    demand_mean: Optional[Union[float, SafetyStockROPInputs]] = None,
    lead_time_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    period_name: Optional[str] = "day",
    # Aliases
    d: Optional[float] = None,
    L: Optional[float] = None,
    sigma_d: Optional[float] = None,
    sigma_L: Optional[float] = None,
    csl: Optional[float] = None,
    z: Optional[float] = None,
    # Old argument names compatibility
    avg_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    lead_time_periods: Optional[float] = None,
    std_lead_time_periods: Optional[float] = None,
    demand_conversion_note: Optional[str] = None,
    lead_time_conversion_note: Optional[str] = None,
) -> ROPResult:
    """
    Calculates Safety Stock (SS) and Reorder Point (ROP).
    Supports all four supply chain variability cases:
      1. Constant demand and constant lead time
      2. Variable demand and constant lead time
      3. Constant demand and variable lead time
      4. Variable demand and variable lead time:
         sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
         SS = z * sigma_LT
         ROP = d * L + SS

    Validates inputs, returns intermediate calculations, final answer,
    and identifies all operational assumptions. Never invents missing numerical inputs.
    """
    if isinstance(demand_mean, SafetyStockROPInputs):
        in_obj = demand_mean
        d_val = in_obj.avg_demand_per_period
        L_val = in_obj.lead_time_periods
        sig_d = in_obj.std_demand_per_period
        sig_L = in_obj.std_lead_time_periods
        sl_val = in_obj.service_level
        z_val = None
        period_str = in_obj.period_name
        demand_conv = in_obj.demand_conversion_note or demand_conversion_note
        lead_time_conv = in_obj.lead_time_conversion_note or lead_time_conversion_note
    else:
        d_val = demand_mean if demand_mean is not None else (avg_demand_per_period if avg_demand_per_period is not None else d)
        L_val = lead_time_mean if lead_time_mean is not None else (lead_time_periods if lead_time_periods is not None else L)
        sig_d = demand_std if demand_std is not None else (std_demand_per_period if std_demand_per_period is not None else sigma_d)
        sig_L = lead_time_std if lead_time_std is not None else (std_lead_time_periods if std_lead_time_periods is not None else sigma_L)
        sl_val = service_level if service_level is not None else csl
        z_val = z_score if z_score is not None else z
        period_str = period_name if period_name is not None else "day"
        demand_conv = demand_conversion_note
        lead_time_conv = lead_time_conversion_note

    # Input validation - never invent missing numerical inputs
    if d_val is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (or 'd') must be provided.")
    if L_val is None:
        raise ValueError("Missing required numerical input: 'lead_time_mean' (or 'L') must be provided.")

    d_val = float(d_val)
    L_val = float(L_val)

    if d_val < 0:
        raise ValueError(f"Demand mean cannot be negative. Got {d_val}")
    if L_val <= 0:
        raise ValueError(f"Lead time mean must be strictly positive (> 0). Got {L_val}")

    sig_d_val = float(sig_d) if sig_d is not None else 0.0
    sig_L_val = float(sig_L) if sig_L is not None else 0.0

    if sig_d_val < 0:
        raise ValueError(f"Demand standard deviation cannot be negative. Got {sig_d_val}")
    if sig_L_val < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative. Got {sig_L_val}")

    is_demand_variable = sig_d_val > 0.0
    is_lead_time_variable = sig_L_val > 0.0
    is_stochastic = is_demand_variable or is_lead_time_variable

    # Determine z-score
    if is_stochastic:
        if z_val is not None:
            calc_z = float(z_val)
            calc_csl = normal_standard_cdf(calc_z) if sl_val is None else float(sl_val)
        elif sl_val is not None:
            calc_csl = float(sl_val)
            if calc_csl <= 0.0 or calc_csl >= 1.0:
                raise ValueError(f"Service level must be strictly between 0 and 1. Got {calc_csl}")
            calc_z = normal_standard_inv_cdf(calc_csl)
        else:
            raise ValueError(
                "Missing required numerical input: Under stochastic demand or lead time, "
                "either 'service_level' (e.g. 0.95) or 'z_score' must be provided."
            )
    else:
        calc_z = 0.0
        calc_csl = 1.0 if sl_val is None else float(sl_val)

    lead_time_demand_mean = d_val * L_val

    # Classify variability case
    if not is_demand_variable and not is_lead_time_variable:
        case_name = "constant_demand_and_constant_lead_time"
        sigma_LT = 0.0
        ss = 0.0
        case_formula = "ROP = d * L (SS = 0)"
        case_assumptions = [
            "Case 1: Constant demand and constant lead time.",
            "Demand rate d is completely deterministic and constant.",
            "Lead time L is completely deterministic and constant.",
            "Safety stock is 0 units because there is zero demand or lead time uncertainty.",
            "Orders arrive exactly after L periods, preventing any stockout if placed at ROP.",
        ]

    elif is_demand_variable and not is_lead_time_variable:
        case_name = "variable_demand_and_constant_lead_time"
        sigma_LT = sig_d_val * math.sqrt(L_val)
        ss = calc_z * sigma_LT
        case_formula = "sigma_LT = sigma_d * sqrt(L); SS = z * sigma_LT; ROP = d * L + SS"
        case_assumptions = [
            "Case 2: Variable demand and constant lead time.",
            "Demand per period is a normally distributed random variable, i.i.d. across periods.",
            "Replenishment lead time L is strictly constant and deterministic.",
            "Cycle service level defines non-stockout probability during the fixed lead time window.",
        ]

    elif not is_demand_variable and is_lead_time_variable:
        case_name = "constant_demand_and_variable_lead_time"
        sigma_LT = d_val * sig_L_val
        ss = calc_z * sigma_LT
        case_formula = "sigma_LT = d * sigma_L; SS = z * sigma_LT; ROP = d * L + SS"
        case_assumptions = [
            "Case 3: Constant demand and variable lead time.",
            "Demand rate d is strictly constant and deterministic.",
            "Replenishment lead time L is a normally distributed random variable with mean L and std dev sigma_L.",
            "Variability during replenishment is driven exclusively by supplier delivery uncertainty.",
        ]

    else:
        case_name = "variable_demand_and_variable_lead_time"
        variance_LT = (L_val * (sig_d_val ** 2)) + ((d_val ** 2) * (sig_L_val ** 2))
        sigma_LT = math.sqrt(variance_LT)
        ss = calc_z * sigma_LT
        case_formula = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2); SS = z * sigma_LT; ROP = d * L + SS"
        case_assumptions = [
            "Case 4: Variable demand and variable lead time.",
            "Demand per period and replenishment lead time are both random variables.",
            "Demand and lead time are mutually independent and uncorrelated.",
            "Total lead time demand variance equals L * sigma_d^2 + d^2 * sigma_L^2.",
            "Lead time demand is approximately normally distributed.",
        ]

    rop = lead_time_demand_mean + ss

    loss_val = normal_loss_function(calc_z) if calc_z > 0 else 0.0
    expected_short = sigma_LT * loss_val
    stockout_risk = (1.0 - calc_csl) if is_stochastic else 0.0

    intermediates = {
        "case_type": case_name,
        "formula": case_formula,
        "d_mean": d_val,
        "L_mean": L_val,
        "sigma_d": sig_d_val,
        "sigma_L": sig_L_val,
        "lead_time_demand_mean_dL": lead_time_demand_mean,
        "sigma_LT": sigma_LT,
        "z_score": calc_z,
        "service_level": calc_csl,
        "safety_stock": ss,
        "reorder_point": rop,
        "normal_loss_Lz": loss_val,
        "expected_units_short_per_cycle": expected_short,
        "stockout_risk_per_cycle": stockout_risk,
        "is_demand_variable": is_demand_variable,
        "is_lead_time_variable": is_lead_time_variable,
    }

    inputs_recorded = {
        "demand_mean": d_val,
        "lead_time_mean": L_val,
        "demand_std": sig_d_val,
        "lead_time_std": sig_L_val,
        "service_level": sl_val,
        "z_score": z_val,
        "period_name": period_str,
    }

    verification = {
        "is_valid": ss >= 0 and rop >= lead_time_demand_mean,
        "z_score_consistent": (abs(normal_standard_cdf(calc_z) - calc_csl) < 1e-4) if is_stochastic else True,
        "positive_safety_stock": ss >= 0,
        "rop_exceeds_lead_time_demand": rop >= lead_time_demand_mean,
        "lead_time_variability_included": sig_L_val > 0,
    }

    audit = _generate_rop_audit(
        d_val=d_val,
        sig_d=sig_d_val,
        L_val=L_val,
        sig_L_val=sig_L_val,
        sl_val=sl_val,
        calc_z=calc_z,
        calc_csl=calc_csl,
        is_stochastic=is_stochastic,
        lead_time_demand_mean=lead_time_demand_mean,
        sigma_LT=sigma_LT,
        ss=ss,
        rop=rop,
        stockout_risk=stockout_risk,
        loss_val=loss_val,
        expected_short=expected_short,
        period_str=period_str,
        case_name=case_name,
        verification=verification,
        demand_conversion_note=demand_conv,
        lead_time_conversion_note=lead_time_conv,
    )

    return ROPResult(
        final_answer=rop,
        reorder_point=rop,
        safety_stock=ss,
        case_type=case_name,
        intermediate_calculations=intermediates,
        assumptions=case_assumptions,
        inputs=inputs_recorded,
        lead_time_demand_mean=lead_time_demand_mean,
        lead_time_demand_std=sigma_LT,
        z_score=calc_z,
        service_level=calc_csl if is_stochastic else None,
        stockout_risk_per_cycle=stockout_risk,
        expected_short_per_cycle=expected_short,
        normal_loss_value=loss_val,
        period_name=period_str,
        verification=verification,
        audit=audit,
    )


# Backwards compatibility alias
calculate_safety_stock_rop = calculate_rop


# ============================================================================
# 4. Fixed-Period Review Model
# ============================================================================

@dataclass
class FixedPeriodReviewInputs:
    review_period: float
    lead_time: float
    avg_demand_per_period: float
    std_demand_per_period: float = 0.0
    service_level: float = 0.95
    current_on_hand: float = 0.0
    current_on_order: float = 0.0
    period_name: str = "day"
    safety_stock: Optional[float] = None
    inventory_position: Optional[float] = None
    backorders: float = 0.0
    demand_conversion_note: Optional[str] = None


@dataclass
class PeriodicReviewResult:
    final_answer: float                 # Recommended order quantity
    recommended_order_quantity: float   # alias for final_answer
    target_inventory: float             # Order-up-to level (S)
    order_up_to_level: float            # alias for target_inventory
    protection_period: float            # R + L
    protection_interval: float          # alias for protection_period
    review_period: float
    lead_time: float
    safety_stock: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    inputs: Dict[str, Any]
    protection_demand_mean: float
    protection_demand_std: float
    z_score: float
    inventory_position: float
    current_inventory_position: float   # alias for inventory_position
    average_cycle_stock: float
    average_total_inventory: float
    comparison_vs_continuous_review_ss: float
    period_name: str
    verification: Dict[str, Any]
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backwards compatibility type alias
FixedPeriodReviewResult = PeriodicReviewResult


def calculate_periodic_review(
    review_period: Optional[Union[float, FixedPeriodReviewInputs]] = None,
    lead_time: Optional[float] = None,
    demand_mean_per_period: Optional[float] = None,
    demand_std_per_period: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None,
    on_hand: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    period_name: Optional[str] = "day",
    # Aliases
    R: Optional[float] = None,
    L: Optional[float] = None,
    d: Optional[float] = None,
    sigma_d: Optional[float] = None,
    csl: Optional[float] = None,
    SS: Optional[float] = None,
    B: Optional[float] = None,
    # Old argument names compatibility
    avg_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    current_on_hand: Optional[float] = None,
    current_on_order: Optional[float] = None,
    demand_conversion_note: Optional[str] = None,
) -> PeriodicReviewResult:
    """
    Calculates Fixed-Period Review (Periodic Review / Order-Up-To) policy:
      Protection period = review period + lead time (R + L)
      Target inventory = demand during protection period + safety stock
      Order quantity = max(0, target inventory - inventory position)

    Validates inputs, returns intermediate calculations, final answer,
    and identifies all operational assumptions. Never invents missing numerical inputs.
    """
    if isinstance(review_period, FixedPeriodReviewInputs):
        in_obj = review_period
        R_val = in_obj.review_period
        L_val = in_obj.lead_time
        d_val = in_obj.avg_demand_per_period
        sig_d_val = in_obj.std_demand_per_period
        sl_val = in_obj.service_level
        z_val = None
        inv_pos_val = getattr(in_obj, "inventory_position", None)
        oh_val = in_obj.current_on_hand
        oo_val = in_obj.current_on_order
        bo_val = getattr(in_obj, "backorders", 0.0)
        ss_val = getattr(in_obj, "safety_stock", None)
        period_str = in_obj.period_name
        demand_conv = getattr(in_obj, "demand_conversion_note", None) or demand_conversion_note
    else:
        R_val = review_period if review_period is not None else R
        L_val = lead_time if lead_time is not None else L
        d_val = demand_mean_per_period if demand_mean_per_period is not None else (avg_demand_per_period if avg_demand_per_period is not None else d)
        sig_d_val = demand_std_per_period if demand_std_per_period is not None else (std_demand_per_period if std_demand_per_period is not None else sigma_d)
        sl_val = service_level if service_level is not None else csl
        z_val = z_score
        inv_pos_val = inventory_position
        oh_val = on_hand if on_hand is not None else current_on_hand
        oo_val = on_order if on_order is not None else current_on_order
        bo_val = backorders if backorders is not None else B
        ss_val = safety_stock if safety_stock is not None else SS
        period_str = period_name if period_name is not None else "day"
        demand_conv = demand_conversion_note

    # Input validation
    if R_val is None:
        raise ValueError("Missing required numerical input: 'review_period' (or 'R') must be provided.")
    if L_val is None:
        raise ValueError("Missing required numerical input: 'lead_time' (or 'L') must be provided.")
    if d_val is None:
        raise ValueError("Missing required numerical input: 'demand_mean_per_period' (or 'd') must be provided.")

    R_val = float(R_val)
    L_val = float(L_val)
    d_val = float(d_val)

    if R_val <= 0:
        raise ValueError(f"Review period must be strictly positive (> 0). Got {R_val}")
    if L_val < 0:
        raise ValueError(f"Lead time cannot be negative. Got {L_val}")
    if d_val < 0:
        raise ValueError(f"Demand mean cannot be negative. Got {d_val}")

    sig_d = float(sig_d_val) if sig_d_val is not None else 0.0
    if sig_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative. Got {sig_d}")

    if ss_val is not None:
        ss = float(ss_val)
        calc_z = ss / (sig_d * math.sqrt(R_val + L_val)) if (sig_d > 0 and (R_val + L_val) > 0) else 0.0
        calc_csl = normal_standard_cdf(calc_z) if calc_z > 0 else 1.0
    elif sig_d > 0:
        if z_val is not None:
            calc_z = float(z_val)
            calc_csl = normal_standard_cdf(calc_z) if sl_val is None else float(sl_val)
        elif sl_val is not None:
            calc_csl = float(sl_val)
            if calc_csl <= 0.0 or calc_csl >= 1.0:
                raise ValueError(f"Service level must be strictly between 0 and 1. Got {calc_csl}")
            calc_z = normal_standard_inv_cdf(calc_csl)
        else:
            raise ValueError(
                "Missing required numerical input: Under demand uncertainty in periodic review, "
                "either 'service_level', 'z_score', or 'safety_stock' must be provided."
            )
        demand_during_protection_std = sig_d * math.sqrt(R_val + L_val)
        ss = calc_z * demand_during_protection_std
    else:
        calc_z = 0.0
        calc_csl = 1.0
        ss = 0.0

    # Protection period = review period + lead time
    protection_period = R_val + L_val
    demand_during_protection_mean = d_val * protection_period
    demand_during_protection_std = sig_d * math.sqrt(protection_period)

    # Target inventory (order-up-to level S)
    target_inventory = demand_during_protection_mean + ss

    # Inventory position = on_hand + on_order - backorders
    bo = float(bo_val) if bo_val is not None else 0.0
    if inv_pos_val is not None:
        inv_pos = float(inv_pos_val)
    elif oh_val is not None or oo_val is not None or bo_val is not None:
        oh = float(oh_val) if oh_val is not None else 0.0
        oo = float(oo_val) if oo_val is not None else 0.0
        inv_pos = oh + oo - bo
    else:
        inv_pos = 0.0

    # Order quantity = max(0, target inventory - inventory position)
    order_quantity = max(0.0, target_inventory - inv_pos)

    avg_cycle_stock = (d_val * R_val) / 2.0
    avg_total_inventory = avg_cycle_stock + ss

    ss_continuous = calc_z * sig_d * math.sqrt(L_val) if L_val > 0 else 0.0
    ss_ratio = ss / ss_continuous if ss_continuous > 0 else float("inf")

    verification = {
        "is_valid": ss >= 0 and target_inventory >= demand_during_protection_mean,
        "protection_interval_equals_R_plus_L": math.isclose(protection_period, R_val + L_val),
        "order_up_to_exceeds_mean_demand": target_inventory >= demand_during_protection_mean,
        "recommended_order_non_negative": order_quantity >= 0.0,
        "safety_stock_ratio_vs_continuous": ss_ratio,
    }

    intermediates = {
        "review_period_R": R_val,
        "lead_time_L": L_val,
        "protection_period_R_plus_L": protection_period,
        "demand_mean_per_period": d_val,
        "demand_std_per_period": sig_d,
        "demand_during_protection_mean": demand_during_protection_mean,
        "demand_during_protection_std": demand_during_protection_std,
        "z_score": calc_z,
        "service_level": calc_csl,
        "safety_stock": ss,
        "target_inventory": target_inventory,
        "inventory_position": inv_pos,
        "unconstrained_order_quantity": target_inventory - inv_pos,
        "recommended_order_quantity": order_quantity,
        "average_cycle_stock": avg_cycle_stock,
        "average_total_inventory": avg_total_inventory,
        "safety_stock_ratio_vs_continuous": ss_ratio,
    }

    assumptions = [
        "Inventory levels are reviewed periodically at fixed intervals of length R.",
        "Replenishment orders are placed only at review epochs and arrive after lead time L.",
        "Protection period against stockout spans the entire review period plus lead time (R + L).",
        "Target inventory (order-up-to level S) restores inventory position to buffer demand across R + L.",
        "Recommended order quantity brings inventory position exactly up to target inventory: q = max(0, S - IP).",
        "Period demand is independent and identically distributed.",
    ]

    inputs_recorded = {
        "review_period": R_val,
        "lead_time": L_val,
        "demand_mean_per_period": d_val,
        "demand_std_per_period": sig_d,
        "service_level": sl_val,
        "inventory_position": inv_pos,
        "period_name": period_str,
    }

    audit = _generate_periodic_review_audit(
        R_val=R_val,
        L_val=L_val,
        protection_period=protection_period,
        d_val=d_val,
        sig_d=sig_d,
        demand_during_protection_mean=demand_during_protection_mean,
        demand_during_protection_std=demand_during_protection_std,
        calc_z=calc_z,
        calc_csl=calc_csl,
        ss=ss,
        target_inventory=target_inventory,
        on_hand=float(oh_val) if oh_val is not None else 0.0,
        on_order=float(oo_val) if oo_val is not None else 0.0,
        backorders=float(bo_val) if bo_val is not None else 0.0,
        inv_pos=inv_pos,
        order_quantity=order_quantity,
        avg_cycle_stock=avg_cycle_stock,
        avg_total_inventory=avg_total_inventory,
        ss_ratio=ss_ratio,
        period_str=period_str,
        verification=verification,
        demand_conversion_note=demand_conv,
    )

    return PeriodicReviewResult(
        final_answer=order_quantity,
        recommended_order_quantity=order_quantity,
        target_inventory=target_inventory,
        order_up_to_level=target_inventory,
        protection_period=protection_period,
        protection_interval=protection_period,
        review_period=R_val,
        lead_time=L_val,
        safety_stock=ss,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        inputs=inputs_recorded,
        protection_demand_mean=demand_during_protection_mean,
        protection_demand_std=demand_during_protection_std,
        z_score=calc_z,
        inventory_position=inv_pos,
        current_inventory_position=inv_pos,
        average_cycle_stock=avg_cycle_stock,
        average_total_inventory=avg_total_inventory,
        comparison_vs_continuous_review_ss=ss_ratio,
        period_name=period_str,
        verification=verification,
        audit=audit,
    )


# Backwards compatibility alias
calculate_fixed_period_review = calculate_periodic_review


# ============================================================================
# 5. Newsvendor Model
# ============================================================================

@dataclass
class NewsvendorInputs:
    unit_cost: float = 0.0
    selling_price: float = 0.0
    salvage_value: float = 0.0
    stockout_penalty: float = 0.0
    mean_demand: float = 0.0
    std_demand: float = 0.0
    underage_cost: Optional[float] = None
    overage_cost: Optional[float] = None
    critical_ratio: Optional[float] = None


@dataclass
class NewsvendorResult:
    final_answer: float                 # Recommended order quantity Q*
    recommended_order_quantity: float   # alias for final_answer
    optimal_order_quantity: float       # alias for final_answer
    underage_cost: float                # Cu
    cost_of_underage: float             # alias for underage_cost
    overage_cost: float                 # Co
    cost_of_overage: float              # alias for overage_cost
    critical_ratio: float               # CR = Cu / (Cu + Co)
    critical_fractile: float            # alias for critical_ratio
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    inputs: Dict[str, Any]
    optimal_z_score: float
    expected_sales: float
    expected_lost_sales: float
    expected_leftover_units: float
    expected_salvage_value: float
    expected_profit: float
    maximum_possible_profit: float
    expected_value_of_perfect_information: float
    verification: Dict[str, Any]
    audit: Optional[AnalystVerificationAudit] = None

    @property
    def analyst_verification_audit(self) -> Optional[AnalystVerificationAudit]:
        return self.audit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    selling_price: Optional[Union[float, NewsvendorInputs]] = None,
    unit_cost: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    salvage_value: Optional[float] = None,
    stockout_penalty: Optional[float] = None,
    critical_ratio: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    # Aliases
    P: Optional[float] = None,
    C: Optional[float] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    V: Optional[float] = None,
    g: Optional[float] = None,
    CR: Optional[float] = None,
    Cu: Optional[float] = None,
    Co: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
) -> NewsvendorResult:
    """
    Calculates the Newsvendor model (Single-Period Stochastic Inventory):
      Underage cost Cu = selling_price - unit_cost + stockout_penalty (or directly given)
      Overage cost Co = unit_cost - salvage_value (or directly given)
      Critical ratio CR = Cu / (Cu + Co)
      Recommended order quantity Q* = demand_mean + z* * demand_std, where Phi(z*) = CR

    Validates inputs, returns intermediate calculations, final answer,
    and identifies all operational assumptions. Never invents missing numerical inputs.
    """
    if isinstance(selling_price, NewsvendorInputs):
        in_obj = selling_price
        p_val = in_obj.selling_price
        c_val = in_obj.unit_cost
        mu_val = in_obj.mean_demand
        sig_val = in_obj.std_demand
        v_val = in_obj.salvage_value
        g_val = in_obj.stockout_penalty
        cr_val = getattr(in_obj, "critical_ratio", None)
        cu_val = getattr(in_obj, "underage_cost", None)
        co_val = getattr(in_obj, "overage_cost", None)
    else:
        p_val = selling_price if selling_price is not None else P
        c_val = unit_cost if unit_cost is not None else C
        mu_val = demand_mean if demand_mean is not None else (mean_demand if mean_demand is not None else mu)
        sig_val = demand_std if demand_std is not None else (std_demand if std_demand is not None else sigma)
        v_val = salvage_value if salvage_value is not None else V
        g_val = stockout_penalty if stockout_penalty is not None else g
        cr_val = critical_ratio if critical_ratio is not None else CR
        cu_val = underage_cost if underage_cost is not None else Cu
        co_val = overage_cost if overage_cost is not None else Co

    # Input validation
    if mu_val is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (or 'mu') must be provided.")
    if sig_val is None:
        raise ValueError("Missing required numerical input: 'demand_std' (or 'sigma') must be provided.")

    mu_val = float(mu_val)
    sig_val = float(sig_val)
    if mu_val < 0:
        raise ValueError(f"Demand mean cannot be negative. Got {mu_val}")
    if sig_val <= 0:
        raise ValueError(f"Demand standard deviation must be strictly positive (> 0). Got {sig_val}")

    # Determine Cu and Co
    if cu_val is not None and co_val is not None:
        cu = float(cu_val)
        co = float(co_val)
        if cu <= 0 or co <= 0:
            raise ValueError(f"Underage cost Cu and Overage cost Co must be strictly positive. Got Cu={cu}, Co={co}")
        p_val = float(p_val) if p_val is not None else (cu + (float(c_val) if c_val is not None else 0.0))
        c_val = float(c_val) if c_val is not None else co
        v_val = float(v_val) if v_val is not None else 0.0
        g_val = float(g_val) if g_val is not None else 0.0
    else:
        if p_val is None:
            raise ValueError("Missing required numerical input: 'selling_price' (or 'P' or 'underage_cost') must be provided.")
        if c_val is None:
            raise ValueError("Missing required numerical input: 'unit_cost' (or 'C' or 'overage_cost') must be provided.")

        p_val = float(p_val)
        c_val = float(c_val)
        v_val = float(v_val) if v_val is not None else 0.0
        g_val = float(g_val) if g_val is not None else 0.0

        if p_val <= 0:
            raise ValueError(f"Selling price must be strictly positive. Got {p_val}")
        if c_val <= 0:
            raise ValueError(f"Unit cost must be strictly positive. Got {c_val}")
        if p_val <= c_val:
            raise ValueError(f"Selling price ({p_val}) must be strictly greater than unit cost ({c_val}).")
        if v_val >= c_val:
            raise ValueError(f"Salvage value ({v_val}) must be strictly less than unit cost ({c_val}).")
        if v_val < 0:
            raise ValueError(f"Salvage value cannot be negative. Got {v_val}")
        if g_val < 0:
            raise ValueError(f"Stockout penalty cannot be negative. Got {g_val}")

        cu = p_val - c_val + g_val
        co = c_val - v_val

    if cu <= 0 or co <= 0:
        raise ValueError(f"Cu and Co must be strictly positive. Got Cu={cu}, Co={co}")

    # Critical ratio
    calc_cr = (cu / (cu + co)) if cr_val is None else float(cr_val)
    if calc_cr <= 0.0 or calc_cr >= 1.0:
        raise ValueError(f"Critical ratio must be strictly between 0 and 1. Got {calc_cr}")

    # Quantile z*
    z_star = normal_standard_inv_cdf(calc_cr)
    q_star = mu_val + (z_star * sig_val)

    # Standard normal unit loss function calculations
    loss_z = normal_loss_function(z_star)
    exp_lost_sales = sig_val * loss_z
    exp_sales = mu_val - exp_lost_sales
    exp_leftover = q_star - exp_sales

    exp_rev = p_val * exp_sales
    exp_procurement = c_val * q_star
    exp_salvage = v_val * exp_leftover
    exp_penalty = g_val * exp_lost_sales
    exp_profit = exp_rev + exp_salvage - exp_procurement - exp_penalty

    max_profit = (p_val - c_val) * mu_val
    evpi = max(0.0, max_profit - exp_profit)

    verification = {
        "is_valid": 0.0 < calc_cr < 1.0 and q_star > 0,
        "critical_fractile_in_bounds": 0.0 < calc_cr < 1.0,
        "inventory_balance_holds": math.isclose(exp_sales + exp_leftover, q_star, rel_tol=1e-3),
        "bias_consistent_with_costs": (q_star > mu_val if cu > co else q_star <= mu_val),
        "evpi_non_negative": evpi >= 0.0,
    }

    intermediates = {
        "selling_price_P": p_val,
        "unit_cost_C": c_val,
        "salvage_value_V": v_val,
        "stockout_penalty_g": g_val,
        "underage_cost_Cu": cu,
        "overage_cost_Co": co,
        "critical_ratio_CR": calc_cr,
        "optimal_z_score": z_star,
        "demand_mean_mu": mu_val,
        "demand_std_sigma": sig_val,
        "safety_buffer_z_sigma": z_star * sig_val,
        "recommended_order_quantity_Q": q_star,
        "normal_unit_loss_Lz": loss_z,
        "expected_sales": exp_sales,
        "expected_lost_sales": exp_lost_sales,
        "expected_leftover_units": exp_leftover,
        "expected_revenue": exp_rev,
        "expected_procurement_cost": exp_procurement,
        "expected_salvage_value": exp_salvage,
        "expected_penalty_cost": exp_penalty,
        "expected_profit": exp_profit,
        "perfect_information_profit": max_profit,
        "evpi": evpi,
    }

    assumptions = [
        "Single selling period for perishable, seasonal, or short-lifecycle items with no replenishment during the period.",
        "Demand is normally distributed with mean mu and standard deviation sigma.",
        "Cost of underage Cu = Price - Cost + Penalty represents the marginal loss per unit of unmet demand.",
        "Cost of overage Co = Cost - Salvage represents the net loss per unit of unsold excess inventory.",
        "Critical ratio CR = Cu / (Cu + Co) establishes the optimal in-stock cycle service probability.",
        "Leftover units are salvaged or liquidated at the specified salvage value.",
    ]

    inputs_recorded = {
        "selling_price": p_val,
        "unit_cost": c_val,
        "demand_mean": mu_val,
        "demand_std": sig_val,
        "salvage_value": v_val,
        "stockout_penalty": g_val,
    }

    audit = _generate_newsvendor_audit(
        p_val=p_val,
        c_val=c_val,
        v_val=v_val,
        g_val=g_val,
        cu=cu,
        co=co,
        calc_cr=calc_cr,
        mu_val=mu_val,
        sig_val=sig_val,
        z_star=z_star,
        q_star=q_star,
        loss_z=loss_z,
        exp_lost_sales=exp_lost_sales,
        exp_sales=exp_sales,
        exp_leftover=exp_leftover,
        exp_rev=exp_rev,
        exp_procurement=exp_procurement,
        exp_salvage=exp_salvage,
        exp_penalty=exp_penalty,
        exp_profit=exp_profit,
        max_profit=max_profit,
        evpi=evpi,
        verification=verification,
    )

    return NewsvendorResult(
        final_answer=q_star,
        recommended_order_quantity=q_star,
        optimal_order_quantity=q_star,
        underage_cost=cu,
        cost_of_underage=cu,
        overage_cost=co,
        cost_of_overage=co,
        critical_ratio=calc_cr,
        critical_fractile=calc_cr,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        inputs=inputs_recorded,
        optimal_z_score=z_star,
        expected_sales=exp_sales,
        expected_lost_sales=exp_lost_sales,
        expected_leftover_units=exp_leftover,
        expected_salvage_value=exp_salvage,
        expected_profit=exp_profit,
        maximum_possible_profit=max_profit,
        expected_value_of_perfect_information=evpi,
        verification=verification,
        audit=audit,
    )


# ============================================================================
# 6. Multicriteria Inventory Classification (Course Stratification Methodology)
# ============================================================================

MODULE_3_POLICIES: Dict[str, Dict[str, str]] = {
    "A": {
        "cycle_count": "Cycle count every week",
        "csl_policy": "High CSL: 95%–98%",
        "description": "Critical items: High dollar velocity or high transaction frequency. Weekly physical audit and tight safety stock control.",
    },
    "B": {
        "cycle_count": "Cycle count once a month",
        "csl_policy": "Medium CSL: 85%–90%",
        "description": "Intermediate items: Moderate dollar volume or moderate frequency. Monthly cycle counts and standard buffer.",
    },
    "C": {
        "cycle_count": "Cycle count once a year",
        "csl_policy": "Low CSL: 70%–80%",
        "description": "Low-velocity items: Infrequent transactions or low dollar volume. Annual physical count.",
    },
    "D": {
        "cycle_count": "Do not count",
        "csl_policy": "No safety stock; donate, destroy, or discount",
        "description": "Obsolete, zero-demand, or dead stock ($0 sales or 0 hits). Liquidation or disposition recommended.",
    },
}

RANK_TO_SCORE: Dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1}
SCORE_TO_RANK: Dict[int, str] = {4: "A", 3: "B", 2: "C", 1: "D"}

MULTICRITERIA_MATRIX_MAP: Dict[Tuple[str, str], str] = {
    # Class A: AA, AB, BA
    ("A", "A"): "A",
    ("A", "B"): "A",
    ("B", "A"): "A",
    # Class B: BB, AC, CA
    ("B", "B"): "B",
    ("A", "C"): "B",
    ("C", "A"): "B",
    # Class C: BC, CB
    ("B", "C"): "C",
    ("C", "B"): "C",
    # Class D: CC, or any item with D in either Sales or Hits ($0 sales or 0 hits)
    ("C", "C"): "D",
    ("A", "D"): "D",
    ("B", "D"): "D",
    ("C", "D"): "D",
    ("D", "D"): "D",
    ("D", "A"): "D",
    ("D", "B"): "D",
    ("D", "C"): "D",
}


def map_multicriteria_class(sales_class: str, hits_class: str) -> str:
    """
    Map (Sales Class, Hits Class) to final Multicriteria Class:
    Class A: AA, AB, BA
    Class B: BB, AC, CA
    Class C: BC, CB
    Class D: CC, or any item with D in either Sales or Hits ($0 sales or 0 hits)
    """
    s = str(sales_class).strip().upper()
    h = str(hits_class).strip().upper()
    if s == "D" or h == "D":
        return "D"
    return MULTICRITERIA_MATRIX_MAP.get((s, h), "D")


def calculate_weighted_score(
    sales_score: int,
    hits_score: int,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> float:
    """
    Compute Combined Weighted Score = (Sales_Weight * Sales_Score) + (Hits_Weight * Hits_Score).
    Default weights: 0.65 for Sales ($) and 0.35 for Hits (per Inventory_Classification_Methodology.pdf).
    Rounded to 6 decimal places to prevent floating-point drift.
    """
    return round(float(sales_weight) * float(sales_score) + float(hits_weight) * float(hits_score), 6)


def score_to_final_class(
    score: float,
    threshold_mode: str = "methodology",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> str:
    """
    Assign Final Inventory Class from Combined Weighted Score.
    Exact thresholds per Inventory_Classification_Methodology.pdf:
    - Class A: Score >= 3.25 (4.00 – 3.25)
    - Class B: Score >= 2.50 (3.24 – 2.50)
    - Class C: Score >= 1.75 (2.49 – 1.75)
    - Class D: Score < 1.75
    Scores are rounded to 6 decimal places to prevent floating-point drift.

    Supports:
    - "methodology" (Default) or "equal_quarter": (3.25, 2.50, 1.75)
    - "alternate": (3.40, 2.80, 2.10)
    - "equal_quarter_legacy": (3.26, 2.51, 1.76)
    - custom_thresholds: (t_a, t_b, t_c)
    """
    if custom_thresholds:
        t_a, t_b, t_c = custom_thresholds
    elif threshold_mode == "alternate":
        t_a, t_b, t_c = 3.40, 2.80, 2.10
    elif threshold_mode == "equal_quarter_legacy":
        t_a, t_b, t_c = 3.26, 2.51, 1.76
    else:  # "methodology", "equal_quarter", or default
        t_a, t_b, t_c = 3.25, 2.50, 1.75

    s = round(float(score), 6)
    if s >= t_a:
        return "A"
    elif s >= t_b:
        return "B"
    elif s >= t_c:
        return "C"
    else:
        return "D"


def assign_criterion_ranks_and_cum(
    values: List[float],
) -> Tuple[List[float], List[str], List[int]]:
    """
    Computes cumulative share (Cum / Total) and assigns individual criterion rank
    using the course brackets:
    - A: cumulative share < 0.60
    - B: 0.60 <= cumulative share < 0.80
    - C: 0.80 <= cumulative share < 0.90
    - D: cumulative share >= 0.90 (or $0 sales / 0 hits)

    Sorted descending, preserving original row order on ties:
    pos_items.sort(key=lambda x: (-x[1], x[0]))

    Returns:
        (cum_shares, ranks, scores) in the original input order.
    """
    n = len(values)
    cum_shares = [1.0] * n
    ranks = ["D"] * n
    scores = [1] * n

    # Filter positive items with their original index
    pos_items = [(i, values[i]) for i in range(n) if values[i] > 0.0]
    # Sort descending by value, preserving original row order on ties
    pos_items.sort(key=lambda x: (-x[1], x[0]))

    total_val = sum(v for _, v in pos_items)
    if total_val <= 0.0:
        return cum_shares, ranks, scores

    running_cum = 0.0
    for order_idx, (orig_idx, val) in enumerate(pos_items):
        running_cum += val
        cum_share = running_cum / total_val
        cum_shares[orig_idx] = cum_share
        if cum_share < 0.60 or order_idx == 0:
            rank = "A"
        elif cum_share < 0.80:
            rank = "B"
        elif cum_share < 0.90:
            rank = "C"
        else:
            rank = "D"
        ranks[orig_idx] = rank
        scores[orig_idx] = RANK_TO_SCORE[rank]

    # Non-positive values (<= 0) are assigned D, score 1, cum_share 1.0
    for i in range(n):
        if values[i] <= 0.0:
            cum_shares[i] = 1.0
            ranks[i] = "D"
            scores[i] = 1

    return cum_shares, ranks, scores


@dataclass
class ClassSummary:
    class_label: str
    item_count: int
    item_pct: float
    total_sales: float
    sales_pct: float
    total_hits: float
    hits_pct: float
    cycle_count_policy: str
    csl_policy: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MulticriteriaClassificationResult:
    enriched_df: Any
    class_summaries: Dict[str, ClassSummary]
    total_items: int
    total_sales: float
    total_hits: float
    summary_table: List[Dict[str, Any]]
    sales_weight: float = 0.65
    hits_weight: float = 0.35
    threshold_mode: str = "methodology"
    thresholds: Tuple[float, float, float] = (3.25, 2.50, 1.75)
    required_summary: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_sales": self.total_sales,
            "total_hits": self.total_hits,
            "sales_weight": self.sales_weight,
            "hits_weight": self.hits_weight,
            "threshold_mode": self.threshold_mode,
            "thresholds": list(self.thresholds),
            "class_summaries": {k: v.to_dict() for k, v in self.class_summaries.items()},
            "summary_table": self.summary_table,
            "required_summary": self.required_summary,
        }


def _clean_numeric_val(val: Any) -> float:
    """Strip currency symbols ($), commas (,), and whitespace from numeric values."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return 0.0
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[\$,\s]", "", val.strip())
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _classify_pareto_prior(
    values: List[float],
    threshold_a: float = 0.80,
    threshold_b: float = 0.95,
) -> List[str]:
    """
    Legacy helper: Assigns Pareto classes based on prior cumulative percentage.
    Provided for backwards-compatibility.
    """
    n = len(values)
    classes = ["D"] * n

    positive_items = [(i, values[i]) for i in range(n) if values[i] > 0.0]
    positive_items.sort(key=lambda x: x[1], reverse=True)

    total_positive = sum(v for _, v in positive_items)
    if total_positive <= 0.0:
        return classes

    running_sum = 0.0
    for idx, val in positive_items:
        prior_cum_pct = running_sum / total_positive
        if prior_cum_pct < threshold_a:
            classes[idx] = "A"
        elif prior_cum_pct < threshold_b:
            classes[idx] = "B"
        else:
            classes[idx] = "C"
        running_sum += val

    return classes


def perform_multicriteria_classification(
    df: Any,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
    threshold_mode: str = "methodology",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
    # Compatibility arguments:
    threshold_a: Optional[float] = None,
    threshold_b: Optional[float] = None,
) -> MulticriteriaClassificationResult:
    """
    Perform Multicriteria Inventory Classification using the exact weighted-score
    stratification methodology from Inventory_Classification_Methodology.pdf and course workbook.

    Procedure:
    1. Parse numeric Sales ($) and Hits (stripping currency $ and formatting commas).
    2. For Sales ($) (sorted descending, preserving original row order on ties): compute
       cumulative sales (Cum) and cumulative share (Cum / Total_Sales), then assign individual Sales_Rank:
       - A: cumulative share < 0.60
       - B: 0.60 <= cumulative share < 0.80
       - C: 0.80 <= cumulative share < 0.90
       - D: cumulative share >= 0.90 (or $0 sales)
    3. For Hits (sorted descending, preserving original row order on ties): compute
       cumulative hits (Cum) and cumulative share (Cum / Total_Hits), then assign individual Hits_Rank using
       the exact same brackets:
       - < 0.60 = A
       - < 0.80 = B
       - < 0.90 = C
       - >= 0.90 = D (or 0 hits)
    4. Rank-to-Score Conversion & Weighted Multicriteria Score:
       Convert each criterion rank into numeric points: A = 4, B = 3, C = 2, D = 1.
       Combined_Weighted_Score = (Sales_Weight * Sales_Score) + (Hits_Weight * Hits_Score)
       Default weights: 0.65 Sales ($) and 0.35 Hits.
    5. Final Multicriteria Class (A, B, C, D) & Module 3 Policies:
       Assign Final_Class from Weighted_Score using methodology thresholds:
       - Class A: Score >= 3.25 (4.00 – 3.25)
       - Class B: Score >= 2.50 (3.24 – 2.50)
       - Class C: Score >= 1.75 (2.49 – 1.75)
       - Class D: Score < 1.75
       (Rounded to 6 decimal places to avoid floating-point drift).
    6. Attach Sales_Cum_Pct, Sales_Rank, Hits_Cum_Pct, Hits_Rank, Sales_Score, Hits_Score,
       Weighted_Score (Combined_Weighted_Score), Final_Class (Final_Inventory_Class),
       Cycle_Count_Policy, and Safety_Stock_CSL_Policy.
    7. Generate Required Classification Summary Table:
       Final Class (A, B, C, D), Number of Items, Total Sales ($) formatted with commas and $.
    """
    import copy

    # Determine column names
    col_names = list(df.columns) if hasattr(df, "columns") else []
    if hasattr(df, "to_dict"):
        records = df.to_dict("records")
    elif isinstance(df, list):
        records = copy.deepcopy(df)
        if records and isinstance(records[0], dict) and not col_names:
            col_names = list(records[0].keys())
    elif isinstance(df, dict):
        if not col_names:
            col_names = list(df.keys())
        n_rows = len(next(iter(df.values()))) if df else 0
        records = [{c: df[c][i] for c in col_names} for i in range(n_rows)]
    else:
        raise ValueError("Unsupported DataFrame type for multicriteria classification.")

    if not records:
        raise ValueError("The provided inventory dataset is empty.")

    # Find sales and hits columns if not specified
    actual_sales_col = sales_col
    actual_hits_col = hits_col

    if not actual_sales_col or actual_sales_col not in col_names:
        for c in col_names:
            norm = re.sub(r"[_\s\-]+", "_", str(c).strip().lower())
            norm_no_sym = re.sub(r"[\$\(\)]", "", norm).strip("_")
            if norm in ["sales", "sales_($)", "sales_$", "sales_dollar", "annual_sales", "revenue", "dollar_sales"] or \
               norm_no_sym in ["sales", "annual_sales", "revenue", "dollar_sales", "total_sales", "sales_usd"]:
                actual_sales_col = c
                break

    if not actual_hits_col or actual_hits_col not in col_names:
        for c in col_names:
            norm = re.sub(r"[_\s\-]+", "_", str(c).strip().lower())
            norm_no_sym = re.sub(r"[\$\(\)]", "", norm).strip("_")
            if norm in ["hits", "picks", "pick_count", "frequency", "order_frequency", "hit_count", "annual_hits"] or \
               norm_no_sym in ["hits", "picks", "pick_count", "frequency", "hit_count", "annual_hits"]:
                actual_hits_col = c
                break

    if not actual_sales_col or not actual_hits_col:
        raise ValueError(
            f"Could not identify Sales and Hits columns in dataset. "
            f"Available columns: {', '.join(col_names)}. "
            f"Expected 'Sales ($)' and 'Hits' (or common synonyms)."
        )

    # Extract and clean numeric values
    sales_values = [_clean_numeric_val(r.get(actual_sales_col)) for r in records]
    hits_values = [_clean_numeric_val(r.get(actual_hits_col)) for r in records]

    # Individual Criterion Ranking using course workbook brackets
    sales_cum, sales_ranks, sales_scores = assign_criterion_ranks_and_cum(sales_values)
    hits_cum, hits_ranks, hits_scores = assign_criterion_ranks_and_cum(hits_values)

    # Weighted Multicriteria Score
    w_sales = float(sales_weight)
    w_hits = float(hits_weight)
    # Ensure weights sum to 1.0 (normalize if not close to 1.0)
    w_sum = w_sales + w_hits
    if abs(w_sum - 1.0) > 1e-4 and w_sum > 0:
        w_sales = w_sales / w_sum
        w_hits = w_hits / w_sum

    weighted_scores = [calculate_weighted_score(s, h, w_sales, w_hits) for s, h in zip(sales_scores, hits_scores)]

    # Final Multicriteria Class
    active_thresholds = custom_thresholds
    if active_thresholds is None:
        if threshold_mode == "alternate":
            active_thresholds = (3.40, 2.80, 2.10)
        elif threshold_mode == "equal_quarter_legacy":
            active_thresholds = (3.26, 2.51, 1.76)
        else:
            active_thresholds = (3.25, 2.50, 1.75)

    final_classes = [
        score_to_final_class(ws, threshold_mode=threshold_mode, custom_thresholds=active_thresholds)
        for ws in weighted_scores
    ]

    cycle_counts = [MODULE_3_POLICIES[c]["cycle_count"] for c in final_classes]
    csl_policies = [MODULE_3_POLICIES[c]["csl_policy"] for c in final_classes]

    # Attach to records
    for i, r in enumerate(records):
        r["Sales_Cum_Pct"] = sales_cum[i]
        r["Sales_Rank"] = sales_ranks[i]
        r["Hits_Cum_Pct"] = hits_cum[i]
        r["Hits_Rank"] = hits_ranks[i]
        r["Sales_Score"] = sales_scores[i]
        r["Hits_Score"] = hits_scores[i]
        r["Weighted_Score"] = weighted_scores[i]
        r["Combined_Weighted_Score"] = weighted_scores[i]
        r["Final_Class"] = final_classes[i]
        r["Final_Inventory_Class"] = final_classes[i]
        r["Cycle_Count_Policy"] = cycle_counts[i]
        r["Safety_Stock_CSL_Policy"] = csl_policies[i]
        # Backwards compatibility aliases
        r["Sales_Class"] = sales_ranks[i]
        r["Hits_Class"] = hits_ranks[i]
        r["Multicriteria_Class"] = final_classes[i]
        r["Recommended_Cycle_Count"] = cycle_counts[i]
        r["Recommended_CSL_Policy"] = csl_policies[i]

    # Attach to original/enriched df if pandas or SimpleDataFrame
    if hasattr(df, "__setitem__"):
        try:
            df["Sales_Cum_Pct"] = sales_cum
            df["Sales_Rank"] = sales_ranks
            df["Hits_Cum_Pct"] = hits_cum
            df["Hits_Rank"] = hits_ranks
            df["Sales_Score"] = sales_scores
            df["Hits_Score"] = hits_scores
            df["Weighted_Score"] = weighted_scores
            df["Combined_Weighted_Score"] = weighted_scores
            df["Final_Class"] = final_classes
            df["Final_Inventory_Class"] = final_classes
            df["Cycle_Count_Policy"] = cycle_counts
            df["Safety_Stock_CSL_Policy"] = csl_policies
            # Aliases
            df["Sales_Class"] = sales_ranks
            df["Hits_Class"] = hits_ranks
            df["Multicriteria_Class"] = final_classes
            df["Recommended_Cycle_Count"] = cycle_counts
            df["Recommended_CSL_Policy"] = csl_policies
            enriched_df = df
        except Exception:
            enriched_df = records
    else:
        enriched_df = records

    # Aggregate summaries across Final_Class
    total_items = len(records)
    total_sales = sum(sales_values)
    total_hits = sum(hits_values)

    class_summaries: Dict[str, ClassSummary] = {}
    summary_table: List[Dict[str, Any]] = []
    required_summary: List[Dict[str, Any]] = []

    for cls_label in ["A", "B", "C", "D"]:
        cls_count = sum(1 for c in final_classes if c == cls_label)
        cls_sales = sum(s for s, c in zip(sales_values, final_classes) if c == cls_label)
        cls_hits = sum(h for h, c in zip(hits_values, final_classes) if c == cls_label)

        item_pct = (cls_count / total_items * 100.0) if total_items > 0 else 0.0
        sales_pct = (cls_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        hits_pct = (cls_hits / total_hits * 100.0) if total_hits > 0 else 0.0

        policy_info = MODULE_3_POLICIES[cls_label]
        summary_obj = ClassSummary(
            class_label=cls_label,
            item_count=cls_count,
            item_pct=item_pct,
            total_sales=cls_sales,
            sales_pct=sales_pct,
            total_hits=cls_hits,
            hits_pct=hits_pct,
            cycle_count_policy=policy_info["cycle_count"],
            csl_policy=policy_info["csl_policy"],
            description=policy_info["description"],
        )
        class_summaries[cls_label] = summary_obj

        # Required Classification Summary (Final Class, Number of Items, Total Sales ($))
        required_summary.append({
            "Final Class": f"Class {cls_label}",
            "Number of Items": cls_count,
            "Total Sales ($)": f"${cls_sales:,.2f}",
        })

        summary_table.append({
            "Class": f"Class {cls_label}",
            "SKU Count": f"{cls_count:,d}",
            "% of SKUs": f"{item_pct:.1f}%",
            "Total Sales ($)": f"${cls_sales:,.2f}",
            "% of Sales": f"{sales_pct:.1f}%",
            "Total Hits": f"{cls_hits:,.0f}",
            "% of Hits": f"{hits_pct:.1f}%",
            "Recommended Cycle Count": policy_info["cycle_count"],
            "Recommended CSL Policy": policy_info["csl_policy"],
        })

    return MulticriteriaClassificationResult(
        enriched_df=enriched_df,
        class_summaries=class_summaries,
        total_items=total_items,
        total_sales=total_sales,
        total_hits=total_hits,
        summary_table=summary_table,
        sales_weight=w_sales,
        hits_weight=w_hits,
        threshold_mode=threshold_mode,
        thresholds=active_thresholds,
        required_summary=required_summary,
    )


