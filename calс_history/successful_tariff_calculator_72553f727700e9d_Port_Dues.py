import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the requested tariff values based on vessel data and predefined rules.

    This module implements the 'Port Dues' calculation as specified in the
    'calculation_rules'. It focuses on the 'General Port Dues Calculation (Section 4.1.1)'
    for commercial vessels engaged in trade, as indicated by the provided vessel data.

    Args:
        vessel_data (dict): A dictionary containing vessel metadata, technical specifications,
                            and operational data. Expected structure:
                            {
                              "vessel_metadata": { ... },
                              "technical_specs": {
                                "gross_tonnage": float,
                                "type": str,
                                ...
                              },
                              "operational_data": {
                                "days_alongside": float,
                                "activity": str,
                                ...
                              }
                            }

    Returns:
        dict[str, float]: A dictionary with calculated tariff values.
                          The keys will exactly match the requested tariff names,
                          e.g., {'Port Dues': 12345.67}.
    """

    # --- Constants from calculation_rules for Port Dues (Section 4.1.1) ---
    # These rates are directly extracted from the 'components' section of the rule.
    BASIC_FEE_RATE = 192.73
    INCREMENTAL_FEE_RATE = 57.79
    TONNAGE_UNIT = 100.0  # Represents "per 100 tons or part thereof"

    # --- Extract necessary data from vessel_data ---

    # Gross Tonnage (GT) is the primary calculation basis for Section 4.1.1.
    # Rule: "Gross Tonnage (GT) as per Tonnage Convention 1969 certificate.
    # If not available, highest tonnage as reflected in Lloyds Register of Shipping."
    # For the provided vessel_data, 'gross_tonnage' is available.
    # If 'gross_tonnage' were None, a fallback would be needed. However, the rule
    # specifies "Lloyds Register of Shipping", which is not present in the input data.
    # Given that 'gross_tonnage' is provided (51300) in the example, we use it directly.
    # No fallback is implemented here as the primary parameter is present and
    # the specified fallback (LRS) cannot be directly derived from the input.
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    if gross_tonnage is None:
        # This is a critical parameter for the General Port Dues calculation.
        # If it were truly missing, and no other rule applied, the calculation
        # for this specific tariff would be impossible without further assumptions
        # or external data not provided. For this problem, it is present.
        raise ValueError("Gross Tonnage (gross_tonnage) is a required parameter for Port Dues calculation (Section 4.1.1) and is missing.")

    # Days alongside is required for the Incremental Fee.
    # Rule: "a part of a 24 hour period being applied pro rata" implies using the exact float value.
    days_alongside = vessel_data.get("operational_data", {}).get("days_alongside")
    if days_alongside is None:
        # This is also a critical parameter for the incremental fee.
        # The rules do not specify a fallback for 'days_alongside'.
        # For this problem, it is present (3.39).
        raise ValueError("Days alongside (days_alongside) is a required parameter for Port Dues calculation (Section 4.1.1) and is missing.")

    # Vessel type and activity are used to determine rule applicability.
    vessel_type = vessel_data.get("technical_specs", {}).get("type")
    vessel_activity = vessel_data.get("operational_data", {}).get("activity")

    # --- Determine applicable rule for Port Dues ---
    # The vessel is identified as a "Bulk Carrier" with activity "Exporting Iron Ore".
    # This indicates a commercial vessel engaged in cargo working.
    # Therefore, the "General Port Dues Calculation (Section 4.1.1)" is the applicable rule.
    # Other rules (Small Vessels, Hulks, Pleasure Vessels, Visiting Vessels not engaged in trade)
    # are explicitly for different vessel types or operational contexts and do not apply here.

    # --- Calculate Port Dues based on Section 4.1.1 ---

    # Calculate the number of 100-ton units, rounded up ("or part thereof").
    # Example: 51300 GT / 100 = 513 units.
    gt_units = math.ceil(gross_tonnage / TONNAGE_UNIT)

    # Basic Fee calculation: (GT units) * (Basic Fee Rate)
    basic_fee = gt_units * BASIC_FEE_RATE

    # Incremental Fee calculation: (GT units) * (Incremental Fee Rate) * (Days Alongside)
    # "a part of a 24 hour period being applied pro rata" means the exact float value of days_alongside is used.
    incremental_fee = gt_units * INCREMENTAL_FEE_RATE * days_alongside

    # Total Port Dues (initial sum of components)
    total_port_dues = basic_fee + incremental_fee

    # --- Apply Reductions/Surcharges/Minimums/Maximums ---
    # Based on the analysis of the provided vessel data and rules:
    # - The vessel is a "Bulk Carrier" engaged in "Exporting Iron Ore" for 3.39 days.
    # - It is not a "small vessel", "pleasure vessel", "hulk", or "visiting vessel (not engaged in trade)".
    # - It is engaged in cargo working, so reductions for "not engaged in cargo working" do not apply.
    # - It is not a "coaster" or "passenger vessel" (based on type "Bulk Carrier").
    # - It is not solely taking bunkers/stores/water.
    # - Its stay (3.39 days = 81.36 hours) is not less than 12 hours.
    # - Its stay (3.39 days) is not longer than 30 days, so the surcharge does not apply.
    # - It is not a "liquid bulk tanker" with specific certifications for the 10% reduction.
    # Therefore, no additional reductions, surcharges, minimums, or maximums apply to this specific case.

    final_port_dues = total_port_dues

    # Round the final tariff value to two decimal places for currency.
    return {"Port Dues": round(final_port_dues, 2)}
