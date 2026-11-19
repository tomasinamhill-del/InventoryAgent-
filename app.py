#!/usr/bin/env python3
"""
Supply Chain Analytics Inventory Decision Agent Application.
Provides an interactive CLI, automated demo mode, and single-query evaluation.

Architecture Workflow:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import Any

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    AgentWorkflowResult,
)


# Terminal ANSI Colors for professional CLI output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
    END = "\033[0m"


def print_banner() -> None:
    banner = f"""{Colors.BOLD}{Colors.CYAN}
================================================================================
          SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT
================================================================================{Colors.END}
{Colors.DIM}Architecture: Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain{Colors.END}
{Colors.DIM}Supported Models:
  [1] Economic Order Quantity (EOQ)
  [2] Quantity Discount Model (All-Units)
  [3] Safety Stock & Reorder Point (Continuous Review)
  [4] Fixed-Period Review Model (Periodic Review / Order-Up-To)
  [5] Newsvendor Model (Single-Period Stochastic){Colors.END}
================================================================================
"""
    print(banner)


def render_workflow_result(result: AgentWorkflowResult) -> None:
    """Formats and prints the full 6-stage agent workflow execution to stdout."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}--- AGENT WORKFLOW TRACE ---{Colors.END}")

    # Stage 1: Question
    print(f"\n{Colors.BOLD}[Stage 1: Customer Question]{Colors.END}")
    print(f"  {Colors.DIM}\"{result.customer_question}\"{Colors.END}")

    # Stage 2: Identify Model
    print(f"\n{Colors.BOLD}[Stage 2: Model Identification]{Colors.END}")
    if result.identified_model != InventoryModelType.UNKNOWN:
        print(f"  Model Identified: {Colors.GREEN}{Colors.BOLD}{result.identified_model.value}{Colors.END}")
        print(f"  Confidence:       {Colors.CYAN}{result.model_confidence*100:.1f}%{Colors.END}")
        print(f"  Rationale:        {result.model_rationale}")
    else:
        print(f"  Model Identified: {Colors.RED}{result.identified_model.value}{Colors.END}")
        print(f"  {result.model_rationale}")

    # Stage 3: Extract Inputs
    print(f"\n{Colors.BOLD}[Stage 3: Extract Inputs]{Colors.END}")
    if result.extracted_inputs:
        for k, v in result.extracted_inputs.items():
            if k == "tiers":
                print(f"  • {Colors.YELLOW}{k}{Colors.END}:")
                for t in v:
                    max_str = f"{t.max_qty:,.0f}" if t.max_qty is not None else "∞"
                    print(f"      - {t.tier_name}: [{t.min_qty:,.0f} to {max_str}] @ ${t.unit_price:,.2f}/unit")
            else:
                print(f"  • {Colors.YELLOW}{k}{Colors.END}: {v}")
    if result.missing_inputs:
        print(f"  {Colors.RED}Missing Required Parameters:{Colors.END} {', '.join(result.missing_inputs)}")

    if result.error_message:
        print(f"\n{Colors.RED}{Colors.BOLD}Error Encountered:{Colors.END} {result.error_message}")
        print("=" * 80)
        return

    # Stage 4: Select Python Tool
    print(f"\n{Colors.BOLD}[Stage 4: Python Tool Selected]{Colors.END}")
    print(f"  Tool Function:    {Colors.BLUE}{Colors.BOLD}{result.tool_selected}(){Colors.END}")

    # Stage 5 & 6: Calculation & Verification
    print(f"\n{Colors.BOLD}[Stage 5 & 6: Numerical Calculation & Verification]{Colors.END}")
    if result.is_verified:
        print(f"  Status:           {Colors.GREEN}{Colors.BOLD}✓ VERIFIED & VALIDATED{Colors.END}")
    else:
        print(f"  Status:           {Colors.YELLOW}{Colors.BOLD}⚠ UNVERIFIED / WARNINGS{Colors.END}")

    for k, v in result.verification_details.items():
        if k != "sensitivity":
            print(f"  • {k}: {v}")

    # Stage 7: Business Explanation
    if result.explanation:
        print(f"\n{Colors.BOLD}{Colors.GREEN}================================================================================{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}                      EXECUTIVE DECISION SUMMARY                                {Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}================================================================================{Colors.END}")
        print(result.explanation.executive_summary)

        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Mathematical & Formula Breakdown ---{Colors.END}")
        for step in result.explanation.formula_trace:
            print(f"  {step}")

        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Managerial Insights & Trade-Offs ---{Colors.END}")
        for insight in result.explanation.managerial_insights:
            print(f"  • {insight}")

        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Verification Summary ---{Colors.END}")
        print(f"  {result.explanation.verification_summary}")

        print(f"\n{Colors.BOLD}{Colors.GREEN}--- Recommended Procurement Action ---{Colors.END}")
        print(f"  ➜ {Colors.BOLD}{result.explanation.recommended_action}{Colors.END}")
        print("=" * 80)


def run_demo(agent: InventoryDecisionAgent) -> None:
    """Executes 5 end-to-end real-world supply chain scenarios."""
    demo_scenarios = [
        (
            "Scenario 1: Classic EOQ with Lead Time",
            "Our annual demand is 12,000 units. Each order costs $50 to place, and holding cost is $2 per unit per year. Unit purchase price is $10. Lead time is 5 days. What is our optimal order quantity and reorder point?"
        ),
        (
            "Scenario 2: All-Units Quantity Discount Schedule",
            "A supplier offers volume discount: 1 to 499 units @ $10.00, 500 to 999 units @ $9.50, and 1000+ units @ $9.00. Annual demand is 4,000 units, order cost is $45, and annual holding rate is 20%. How much should we order to minimize total annual cost?"
        ),
        (
            "Scenario 3: Continuous Review Safety Stock & Reorder Point",
            "Daily demand averages 100 units with standard deviation of 15 units. Supplier lead time is 4 days. We need a 95% cycle service level. What should our safety stock and reorder point be?"
        ),
        (
            "Scenario 4: Fixed-Period Review (Periodic Order-Up-To)",
            "We review inventory every 14 days. Lead time is 7 days. Average demand is 50 units with std dev 8. Target service level is 98%. On hand we have 420 units and 100 on order. What is the order-up-to level and how much should we order?"
        ),
        (
            "Scenario 5: Seasonal Perishable Item (Newsvendor Model)",
            "We sell winter jackets for $250 each. We buy them for $100. Any unsold jackets at the end of the season are sold at markdown for $40. Demand is normally distributed with mean 500 and std dev 80. How many jackets should we purchase?"
        ),
    ]

    print(f"\n{Colors.BOLD}{Colors.YELLOW}RUNNING AUTOMATED DEMONSTRATION OF ALL 5 INVENTORY MODELS...{Colors.END}\n")

    for title, query in demo_scenarios:
        print(f"\n{Colors.BOLD}{Colors.HEADER}>>> {title.upper()} <<<{Colors.END}")
        res = agent.process_query(query)
        render_workflow_result(res)
        print("\n" + "#" * 80 + "\n")


def interactive_mode(agent: InventoryDecisionAgent) -> None:
    """Interactive Command Line REPL for natural-language questions."""
    print(f"\n{Colors.BOLD}Interactive Mode Active.{Colors.END}")
    print(f"Ask any natural-language inventory question, or type {Colors.YELLOW}'demo'{Colors.END}, {Colors.YELLOW}'examples'{Colors.END}, or {Colors.YELLOW}'exit'{Colors.END}.\n")

    examples = [
        "1. EOQ: 'Annual demand is 10,000 units, ordering cost is $60, and annual holding cost is $3 per unit. What is the EOQ?'",
        "2. Discount: 'Demand is 5,000 units, order cost is $50, holding rate is 25%. Price is $20 for 1-999, and $18 for 1000+. Optimal order?'",
        "3. Safety Stock: 'Average daily demand is 80 units with std dev 12. Lead time is 5 days. Target service level 99%. What is ROP?'",
        "4. Periodic: 'Review period is 7 days, lead time is 3 days, daily demand is 40 with std 6, service level 95%, on hand 150. Order size?'",
        "5. Newsvendor: 'Selling price is $80, cost is $30, salvage value is $10. Mean demand is 1,200 with std dev 200. Optimal quantity?'",
    ]

    while True:
        try:
            prompt = input(f"{Colors.BOLD}{Colors.CYAN}inventory-agent > {Colors.END}").strip()
            if not prompt:
                continue

            cmd = prompt.lower()
            if cmd in ("exit", "quit", "q"):
                print(f"{Colors.GREEN}Exiting Inventory Decision Agent. Goodbye!{Colors.END}")
                break

            if cmd == "demo":
                run_demo(agent)
                continue

            if cmd == "examples":
                print(f"\n{Colors.BOLD}Sample Questions You Can Ask:{Colors.END}")
                for ex in examples:
                    print(f"  • {ex}")
                print()
                continue

            # Process query
            result = agent.process_query(prompt)
            render_workflow_result(result)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.GREEN}Session interrupted. Goodbye!{Colors.END}")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent CLI"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run automated demonstration across all 5 models"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Process a single natural-language query and exit"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output result as JSON (useful for API integrations)"
    )

    args = parser.parse_args()
    agent = InventoryDecisionAgent()

    if args.demo:
        print_banner()
        run_demo(agent)
    elif args.query:
        result = agent.process_query(args.query)
        if args.json:
            # Serializer helper for dataclasses / enums
            out = {
                "question": result.customer_question,
                "identified_model": result.identified_model.value,
                "confidence": result.model_confidence,
                "rationale": result.model_rationale,
                "extracted_inputs": {
                    k: (str(v) if k == "tiers" else v) for k, v in result.extracted_inputs.items()
                },
                "missing_inputs": result.missing_inputs,
                "tool_selected": result.tool_selected,
                "is_verified": result.is_verified,
                "error_message": result.error_message,
                "explanation": (
                    {
                        "executive_summary": result.explanation.executive_summary,
                        "formula_trace": result.explanation.formula_trace,
                        "managerial_insights": result.explanation.managerial_insights,
                        "verification_summary": result.explanation.verification_summary,
                        "recommended_action": result.explanation.recommended_action,
                    }
                    if result.explanation
                    else None
                ),
            }
            print(json.dumps(out, indent=2))
        else:
            render_workflow_result(result)
    else:
        print_banner()
        interactive_mode(agent)


if __name__ == "__main__":
    main()
