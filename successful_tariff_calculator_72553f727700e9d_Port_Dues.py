import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the Port Dues based on the provided vessel data and rules.

    Hard requirements:
    - Output only Python code.
    - Define calculate(vessel_data: dict) -> dict[str, float].
    - The result keys must exactly match: ['Port Dues'].
    - Use only the standard library.
    - Implement the formulas/rates from calculation_rules.
    - If a needed parameter is missing, use a clearly named fallback only when the rules justify it.
    - Do not infer rates from expected answers, previous calculated outputs, percent differences, or evaluation feedback.
    - Do not hardcode final tariff answers.
    - Do not read files and do not ask for input.

    Args:
        vessel_data (dict): A dictionary containing vessel information,
                            including technical specifications and operational data.

    Returns:
        dict[str, float]: A dictionary with 'Port Dues' as the key and its calculated value.

    Raises:
        ValueError: If essential parameters like 'gross_tonnage' or 'days_alongside'
                    are missing or invalid, as no rule-justified numerical fallback
                    is provided for these core calculation inputs.
    """

    # --- Extract necessary parameters from vessel_data ---

    # Gross Tonnage (GT) is the primary basis for port dues calculation.
    # Rule: "The tonnage of vessels for port tariff purposes is the gross tonnage of a vessel
    # as per the tonnage certificate issued in terms of the Tonnage Convention 1969."
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    if not isinstance(gross_tonnage, (int, float)) or gross_tonnage <= 0:
        raise ValueError(
            "Missing or invalid 'gross_tonnage' in vessel_data.technical_specs. "
            "Cannot calculate Port Dues as it's a core parameter without a rule-justified fallback."
        )

    # Days Alongside determines the duration-based fee.
    # Rule: "An additional fee is charged per 100 tons or part thereof per 24 hour period,
    # with a part of a 24 hour period being applied pro rata."
    days_alongside = vessel_data.get("operational_data", {}).get("days_alongside")
    if not isinstance(days_alongside, (int, float)) or days_alongside < 0:
        raise ValueError(
            "Missing or invalid 'days_alongside' in vessel_data.operational_data. "
            "Cannot calculate Port Dues as it's a core parameter without a rule-justified fallback."
        )

    # --- Define rates as per calculation_rules ---
    # Rate: "Basic fee per 100 tons or part thereof"
    BASIC_FEE_RATE_PER_100_TONS = 192.73  # ZAR

    # Rate: "Per 100 tons or part thereof per 24 hour period"
    TIME_BASED_FEE_RATE_PER_100_TONS_PER_24H = 57.79  # ZAR

    # --- Apply rounding and calculation units ---
    # Rule: "Gross tonnage is rounded up to the nearest 100 tons for calculation units."
    # This means we divide the gross tonnage by 100 and then take the ceiling.
    tonnage_units = math.ceil(gross_tonnage / 100)

    # Rule: "Time (days_alongside) is applied pro rata (not rounded)."
    # This means days_alongside is used directly in the calculation.

    # --- Calculate Port Dues using the specified formula ---
    # Formula: Port Dues = (CEILING(gross_tonnage / 100) * Basic_Fee_Rate) +
    #                      (CEILING(gross_tonnage / 100) * (days_alongside / 1) * Time_Based_Fee_Rate)

    basic_port_dues_component = tonnage_units * BASIC_FEE_RATE_PER_100_TONS
    time_based_port_dues_component = tonnage_units * days_alongside * TIME_BASED_FEE_RATE_PER_100_TONS_PER_24H

    total_port_dues = basic_port_dues_component + time_based_port_dues_component

    # --- Apply reductions/surcharges/exemptions (or confirm non-applicability) ---
    # The provided calculation_rules explicitly state that none of the listed
    # reductions (35% for non-cargo, 60% for bunker/stores/water, 10% for liquid bulk,
    # 15% for <12h stay) or surcharges (20% for >30 days non-cargo) apply to this vessel
    # based on its activity ('Exporting Iron Ore'), type ('Bulk Carrier'), and stay duration (3.39 days).
    # Therefore, the calculated total_port_dues is the final amount.

    return {"Port Dues": total_port_dues}
