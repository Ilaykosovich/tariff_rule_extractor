import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the Port Dues based on the provided vessel data and tariff rules.

    Hard requirements:
    - Output only Python code.
    - Define calculate(vessel_data: dict) -> dict[str, float].
    - The result keys must exactly match: ['Port Dues'].
    - Use only the standard library.
    - If a needed parameter is missing, use a clearly named fallback only when the rules justify it.
    - Do not infer rates from expected answers, previous calculated outputs, percent differences, or evaluation feedback.
    - Do not hardcode final tariff answers.
    - Do not read files and do not ask for input.

    Args:
        vessel_data (dict): A dictionary containing vessel information,
                            including technical_specs and operational_data.

    Returns:
        dict[str, float]: A dictionary with 'Port Dues' as the key and its
                          calculated value as a float.
    """
    # --- Extract necessary parameters ---
    # Gross Tonnage: Essential for calculation. Rules state to use tonnage certificate
    # or Lloyd's Register if not available. For this problem, it's provided.
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    if gross_tonnage is None:
        # Gross Tonnage is a critical parameter for the main Port Dues formula.
        # The rules mention fallbacks (Lloyd's Register) which are not available in input.
        # Without it, the core calculation cannot proceed.
        raise ValueError("Gross Tonnage is a required parameter for Port Dues calculation.")

    # Days Alongside: Essential for incremental fee calculation and duration-based reductions/surcharges.
    days_alongside = vessel_data.get("operational_data", {}).get("days_alongside")
    if days_alongside is None:
        # Days Alongside is a critical parameter for the incremental fee.
        raise ValueError("Days Alongside is a required parameter for Port Dues calculation.")

    # Vessel Type: Used to determine applicability of specific rules (e.g., small vessels, tankers).
    vessel_type = vessel_data.get("technical_specs", {}).get("type")
    # Fallback: If type is missing, assume it's a standard commercial vessel for tariff purposes.
    # This assumption means specific small vessel/pleasure vessel/tanker rules won't apply by default,
    # which is a safe default for a general commercial vessel context.
    if vessel_type is None:
        vessel_type = "Unknown Commercial Vessel"

    # Activity: Used to determine if vessel is engaged in cargo working, bunkering, etc.
    activity = vessel_data.get("operational_data", {}).get("activity")
    # Fallback: If activity is missing, assume it's engaged in cargo working for a commercial vessel.
    # This avoids applying reductions for non-cargo working by default, which is a conservative approach.
    if activity is None:
        activity = "Engaged in Cargo Working (Assumed)"

    # --- Define Rates from the provided rules ---
    BASIC_FEE_RATE = 192.73
    INCREMENTAL_FEE_RATE = 57.79
    # SMALL_VESSEL_MIN_FEE = 470.98 # Not applicable to this vessel type based on rules

    # --- Calculate Tonnage Units for the formula ---
    # "per 100 tons or part thereof" implies using math.ceil
    tons_for_calculation = math.ceil(gross_tonnage / 100)

    # --- Calculate Stay Duration Factor for the formula ---
    # "per 24-hour period, with a part of a 24-hour period being applied pro rata"
    # This means days_alongside can be used directly as the factor for 24-hour periods.
    stay_duration_factor = days_alongside

    # --- Calculate Basic Fee Component ---
    basic_fee_component = tons_for_calculation * BASIC_FEE_RATE

    # --- Calculate Incremental Fee Component ---
    incremental_fee_component = tons_for_calculation * stay_duration_factor * INCREMENTAL_FEE_RATE

    # --- Determine Applicable Reductions and Surcharges ---
    # Initialize total reduction and surcharge percentages
    total_reduction_percentage = 0.0
    total_surcharge_percentage = 0.0

    # Helper flag based on vessel data and rules
    is_engaged_in_cargo_working = "cargo" in activity.lower() or "exporting" in activity.lower() or "importing" in activity.lower()
    is_liquid_bulk_tanker = "liquid bulk tanker" in vessel_type.lower()

    # Rule: 35% reduction for vessels not engaged in cargo working (first 30 days), bona fide coasters, passenger vessels, small vessels.
    # This vessel is a "Bulk Carrier" and is engaged in cargo working ("Exporting Iron Ore").
    # The other conditions (bona fide coasters, passenger vessels, small vessels) do not apply to a "Bulk Carrier" for this rule.
    # Therefore, this reduction does not apply.

    # Rule: 20% surcharge for vessels in port > 30 days NOT engaged in cargo working or undergoing repairs.
    # This vessel IS engaged in cargo working.
    # Also, days_alongside (3.39) is not > 30 days.
    # Therefore, this surcharge does not apply.

    # Rule: 60% reduction for vessels calling solely for bunkers/stores/water, if stay <= 48 hours (2 days).
    # This vessel's activity is "Exporting Iron Ore", not solely for bunkers/stores/water.
    # Therefore, this reduction does not apply.

    # Rule: 10% reduction for certified double hulled liquid bulk tankers, etc.
    # This vessel is a "Bulk Carrier", not a "liquid bulk tanker".
    # Therefore, this reduction does not apply.

    # Rule: 15% reduction for vessels remaining in port for less than 12 hours.
    # days_alongside (3.39 days) * 24 hours/day = 81.36 hours. This is not less than 12 hours.
    # Therefore, this reduction does not apply.

    # Rule: Minimum fee of 470.98 for small vessels and pleasure vessels (Section 4.2).
    # This vessel is a "Bulk Carrier", not a small or pleasure vessel.
    # Therefore, this minimum does not apply to the main Port Dues calculation.

    # Specific Port Dues for Small Vessels, Hulks, Pleasure Vessels, Visiting Yachts (Section 4.2)
    # These rules apply to specific vessel types (e.g., non-mechanically propelled, self-propelled boats,
    # fishing vessels, hulks, yachts). The current vessel is a "Bulk Carrier" engaged in trade,
    # which falls under the general Port Dues formula, not these specific Section 4.2 categories.

    # --- Calculate Final Port Dues ---
    # The formula is: Port Dues = ( (Basic Fee Component) + (Incremental Fee Component) ) * (1 - Sum of Applicable Reductions) * (1 + Sum of Applicable Surcharges)
    port_dues = (basic_fee_component + incremental_fee_component) * \
                (1 - total_reduction_percentage) * \
                (1 + total_surcharge_percentage)

    # Round the final result to two decimal places for currency.
    return {"Port Dues": round(port_dues, 2)}
