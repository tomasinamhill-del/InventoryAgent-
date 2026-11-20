# Project Bug-Fix Validation Notes

## Bug-Fix 2026-11-19: Currency-Formatted String Parsing in CSV Ingestion

### 1. Summary
- **Module Affected**: [`streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/streamlit_app.py)
- **Functions Modified**: `load_and_process_inventory_csv()`, `clean_numeric_series()`, `SimpleSeries.apply()`
- **Tests Added**: `TestCSVInventoryAnalyticsDashboard.test_clean_numeric_series_currency_commas_whitespace`, `TestCSVInventoryAnalyticsDashboard.test_load_and_process_inventory_csv_currency_formatted_prices_and_commas` in [`tests/test_streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_streamlit_app.py)
- **Issue Classification**: Data Parsing / Numeric Coercion Defect (Silent Zero Valuation)

---

### 2. Defect Description & Impact
When users uploaded enterprise inventory CSV files where numeric columns contained formatted monetary values (e.g., `"$2.50"`, `"$1,250.00"`) or comma-separated unit counts (e.g., `"1,000"`), passing the raw string Series directly to `pd.to_numeric(series, errors="coerce")` caused pandas to fail parsing. The values were converted to `NaN`, and subsequent `.fillna(0.0)` replaced them with `0.0`.

#### Consequence:
- Formatted unit costs (e.g. `"$1,250.00"`) became `0.0`.
- Line-item inventory valuation `inventory_value = units * unit_cost` evaluated to `0.0`.
- Key dashboard metrics (Total Inventory Value, Average Unit Cost, Highest Unit Cost) were completely distorted and showed zero or severely underestimated totals.

---

### 3. Root Cause Analysis
Standard `pandas.to_numeric(..., errors="coerce")` parses clean numerical strings (e.g., `"2.50"`, `"1250.0"`) and native numeric types (`float`, `int`). It does not strip currency symbols (`$`), thousands grouping commas (`,`), or whitespace. When passed values with those formatting characters, pandas treats the string as invalid numeric data and coerces it to `NaN`.

---

### 4. Fix Implementation

1. **Created Sanitization Helper `clean_numeric_series()`**:
   ```python
   def clean_numeric_series(series: Any) -> Any:
       """
       Sanitize string values in a numeric column by stripping currency symbols ($),
       commas (,), and whitespace before calling pd.to_numeric(..., errors="coerce").
       """
       def _strip_val(x: Any) -> Any:
           if x is None or (isinstance(x, float) and math.isnan(x)):
               return x
           if isinstance(x, str):
               # Remove currency symbols ($), thousand-separator commas (,), and whitespace
               return re.sub(r"[\$,\s]", "", x.strip())
           return x

       if hasattr(series, "apply"):
           return series.apply(_strip_val)
       elif hasattr(series, "data"):
           return SimpleSeries([_strip_val(x) for x in series.data], name=getattr(series, "name", ""))
       elif isinstance(series, (list, tuple)):
           return [_strip_val(x) for x in series]
       return series
   ```

2. **Updated `load_and_process_inventory_csv()` in [`streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/streamlit_app.py)**:
   Pre-processes both `units` and `unit_cost` columns with `clean_numeric_series()` prior to invoking `pd.to_numeric(..., errors="coerce")`:
   ```python
   cleaned_units = clean_numeric_series(processed_df["units"])
   processed_df["units"] = pd.to_numeric(cleaned_units, errors="coerce")
   if hasattr(processed_df["units"], "fillna"):
       processed_df["units"] = processed_df["units"].fillna(0)

   cleaned_cost = clean_numeric_series(processed_df["unit_cost"])
   processed_df["unit_cost"] = pd.to_numeric(cleaned_cost, errors="coerce")
   if hasattr(processed_df["unit_cost"], "fillna"):
       processed_df["unit_cost"] = processed_df["unit_cost"].fillna(0.0)
   ```

3. **Added `apply()` Method to `SimpleSeries`**:
   Ensures that both standard pandas DataFrames and the lightweight `SimplePandasShim` compatibility layer support `.apply()` interchangeably across testing and production environments.

---

### 5. Validation & Automated Test Results
Added two dedicated regression tests in [`tests/test_streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_streamlit_app.py):
- `test_clean_numeric_series_currency_commas_whitespace`: Verifies that `"$2.50"`, `"$1,250.00"`, `"1,000"`, and `" 25,500.50 "` are accurately stripped to `"2.50"`, `"1250.00"`, `"1000"`, and `"25500.50"`.
- `test_load_and_process_inventory_csv_currency_formatted_prices_and_commas`: Ingests a CSV with `units="1,000"`, `unit_cost="$2.50"`, and `unit_cost="$1,250.00"`, and asserts:
  - Unit counts parse to `1000.0` and `250.0` (not `0.0`).
  - Unit costs parse to `2.50` and `1250.00` (not `0.0`).
  - Line-item `inventory_value` evaluates to `$2,500.00` and `$312,500.00`.
  - Total inventory valuation accurately reflects `$318,750.00`.

**Test Suite Execution**:
```bash
./run.sh test
# 47/47 tests passing (0.010s)
```


## Enhancement / Methodology Alignment 2026-11-19: Multicriteria Classification Defaults & Exact Thresholds

### 1. Summary
- **Modules Affected**: [`inventory_tools.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/inventory_tools.py), [`streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/streamlit_app.py), [`README.md`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/README.md)
- **Functions Modified**: `assign_criterion_ranks_and_cum()`, `calculate_weighted_score()`, `score_to_final_class()`, `perform_multicriteria_classification()`, `load_and_process_inventory_csv()`, `render_csv_analytics_dashboard()`
- **Tests Updated & Added**: [`tests/test_streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_streamlit_app.py) (`test_rank_to_score_and_weighted_score_calculation`, `test_score_to_final_class_threshold_sets`, `test_load_and_process_classification_sample_csv`, `test_tie_breaking_preserves_original_row_order`, `test_weight_slider_tuning_and_alternate_scale`)
- **Reference Standard**: `Inventory_Classification_Methodology.pdf`

---

### 2. Implementation Details

1. **Criterion Ranking & Tie-Breaking**:
   - Ranked Sales ($) and Hits separately using cumulative volume share brackets:
     - $< 0.60 \rightarrow \mathbf{A}$ (score 4)
     - $0.60 \text{ to } < 0.80 \rightarrow \mathbf{B}$ (score 3)
     - $0.80 \text{ to } < 0.90 \rightarrow \mathbf{C}$ (score 2)
     - $\ge 0.90 \rightarrow \mathbf{D}$ (score 1)
   - Tie-breaking strictly preserves original row order on ties via `pos_items.sort(key=lambda x: (-x[1], x[0]))`.
   - Items with $\$0.00$ sales or $0$ hits are automatically assigned Class D (score 1, cumulative share 1.0).

2. **Default Weights**:
   - Default weights set to **0.65 for Sales** and **0.35 for Hits**:
     $$\text{Combined Weighted Score} = 0.65 \times \text{Sales\_Score} + 0.35 \times \text{Hits\_Score}$$

3. **Final Inventory Class Exact Thresholds**:
   - Class A: $\ge 3.25$ ($4.00 - 3.25$)
   - Class B: $\ge 2.50$ ($3.24 - 2.50$)
   - Class C: $\ge 1.75$ ($2.49 - 1.75$)
   - Class D: $< 1.75$
   - Scores are rounded to 6 decimal places (`round(float(score), 6)`) to eliminate floating-point drift.

4. **Required Classification Summary Table**:
   - Structured summary table generated and rendered prominently in Streamlit with columns:
     - `Final Class` (`Class A`, `Class B`, `Class C`, `Class D`)
     - `Number of Items`
     - `Total Sales ($)` (formatted with commas and dollar signs, e.g., `\$190,000.00`)


## Bug-Fix 2026-11-19: Natural-Language Demand & Holding Cost Time-Unit Normalization

### 1. Summary
- **Modules Affected**: [`inventory_agent.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/inventory_agent.py), [`inventory_tools.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/inventory_tools.py), [`streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/streamlit_app.py)
- **Functions Modified**: `_extract_demand_with_time_unit()`, `_extract_holding_cost()`, `_generate_eoq_audit()`, `_generate_quantity_discount_audit()`
- **Tests Added**: `test_agent_eoq_query_with_monthly_demand`, `test_demand_time_unit_conversions`, `test_holding_cost_time_unit_conversions`, `test_monthly_demand_eoq_audit_and_ui_rendering`
- **Issue Classification**: Natural Language Entity Extraction / Time-Unit Mismatch Defect

### 2. Defect Description & Impact
When queries specified demand rates with non-annual time units (e.g., `"Demand is 1,000 units per month"` or `"...per week"`, `"...per day"`), naive extraction took the literal number (`1000.0`) as annual demand instead of normalizing to an annual basis (`1,000 * 12 = 12,000 units/year`). This caused severe underestimation of EOQ, total costs, and inventory buffers.

### 3. Fix Implementation
- Implemented `_extract_demand_with_time_unit()` supporting prefix and postfix time units (monthly $\times 12$, weekly $\times 52$, daily $\times$ work days [default 250, or 365 for calendar days], quarterly $\times 4$).
- Implemented `_extract_holding_cost()` with support for monthly, weekly, and daily holding cost rates.
- Generated explicit unit conversion notes in Section 1 and Section 7 of the Analyst Verification Audit (e.g., `1,000 units/month × 12 = 12,000 units/year`).


## Bug-Fix 2026-11-19: Safety Stock & ROP Lead-Time Std Dev Extraction & Unit Alignment

### 1. Summary
- **Modules Affected**: [`inventory_agent.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/inventory_agent.py), [`inventory_tools.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/inventory_tools.py), [`streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/streamlit_app.py)
- **Functions Modified**: `_extract_lead_time_parameters()`, `_extract_demand_parameters()`, `extract_inputs()`, `select_and_calculate_tool()`, `explain_result()`, `_generate_rop_audit()`, `calculate_rop()`
- **Tests Added**:
  - `test_agent_safety_stock_rop_test_7_variable_demand_and_lead_time` in [`tests/test_inventory_tools.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_inventory_tools.py)
  - `test_lead_time_std_dev_phrasings_and_unit_alignment` in [`tests/test_inventory_tools.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_inventory_tools.py)
  - `test_rop_lead_time_variability_and_unit_alignment_audit_and_ui_rendering` in [`tests/test_streamlit_app.py`](file:///Users/tomasinahill/Downloads/Inventory%20Agent/tests/test_streamlit_app.py)
- **Issue Classification**: Natural Language Entity Extraction & Unit Alignment Defect

### 2. Defect Description & Impact
1. **Lead Time Std Dev Extraction Failure**:
   Natural language queries containing compound sentences like `"Average lead time is 3 weeks with a standard deviation of 0.5 weeks"` or `"lead time std dev of 0.5"` failed to extract $\sigma_L$. The parameter defaulted to $0.0$, falsely degrading the model from Case 4 (*Variable Demand and Variable Lead Time*) to Case 2 (*Variable Demand and Constant Lead Time*).
2. **Unit Mismatch in $\mu_{DLT}$ and $\sigma_{DLT}$**:
   When demand was stated per day (e.g., $40 \text{ units/day}$) and lead time in weeks (e.g., $3 \text{ weeks}$), the engine multiplied $40 \times 3 = 120$ units directly instead of converting lead time to days ($3 \times 7 = 21$ days $\implies \mu_{DLT} = 40 \times 21 = 840$ units).

### 3. Fix Implementation
1. **Lead Time Parameter & Std Dev Extraction**:
   - Upgraded `_extract_lead_time_parameters()` to parse compound phrases (`"lead time is 3 weeks with a standard deviation of 0.5 weeks"`, `"lead time averages 3 weeks (std dev 0.5 weeks)"`), inverted patterns (`"standard deviation of lead time is 0.5"`), and standalone std dev expressions.
   - Isolated period unit detection specifically to the lead-time sentence (`week`, `day`, `month`).
   - Upgraded `_extract_demand_parameters()` with lead-time clause masking to prevent lead-time std dev and period tokens from leaking into demand parameters.
2. **Automatic Period Unit Alignment**:
   - When demand is daily (`"day"`) and lead time is weekly (`"week"`), converts lead time to calendar days ($L = L \times 7.0$ and $\sigma_L = \sigma_L \times 7.0$), sets `period_name = "day"`, and logs an explicit alignment note:
     `"Converted 3.0 weeks lead time to 21.0 days and 0.50 weeks std dev to 3.50 days (at 7 calendar days/week; at 5 business days/week: 15.0 days, std dev 2.50 days)"`.
   - Supports conversions for months-to-days and days-to-weeks.
   - Propagated `lead_time_conversion_note` through `SafetyStockROPInputs`, `calculate_rop()`, `_generate_rop_audit()`, and the Streamlit parameter table.

### 4. Verification with Exact Test 7 Prompt
- **Prompt**:
  > *"A product has average demand of 40 units per day with a standard deviation of 8 units per day. Average lead time is 3 weeks with a standard deviation of 0.5 weeks. Determine the safety stock and reorder point for a 95% service level."*
- **Mathematical Derivation & Audit Output**:
  - $\bar{d} = 40.0 \text{ units/day}, \quad \sigma_d = 8.0 \text{ units/day}$
  - $\bar{L} = 3 \text{ weeks} \times 7 = 21.0 \text{ days}, \quad \sigma_L = 0.5 \text{ weeks} \times 7 = 3.5 \text{ days}$
  - $\mu_{DLT} = \bar{d} \times \bar{L} = 40.0 \times 21.0 = 840.0 \text{ units}$
  - $\text{Var}(DLT) = \bar{L} \sigma_d^2 + \bar{d}^2 \sigma_L^2 = 21(64) + 1600(12.25) = 1,344 + 19,600 = 20,944$
  - $\sigma_{DLT} = \sqrt{20,944} \approx 144.7204 \text{ units}$
  - $z = \Phi^{-1}(0.95) \approx 1.6448536$
  - $\text{Safety Stock (SS)} = z \times \sigma_{DLT} = 1.6448536 \times 144.7204 \approx 238.04 \approx \mathbf{238.0 \text{ units}}$
  - $\text{Reorder Point (ROP)} = \mu_{DLT} + \text{SS} = 840.0 + 238.04 \approx 1,078.04 \approx \mathbf{1,078.0 \text{ units}}$
  - $\text{Operational Case} = \mathbf{variable\_demand\_and\_variable\_lead\_time}$
  - Verification Invariant $\text{lead\_time\_variability\_included} = \mathbf{PASS \ (True)}$
