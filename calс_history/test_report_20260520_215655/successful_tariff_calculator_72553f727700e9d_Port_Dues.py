import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the requested tariff values based on vessel data and predefined rules.

    Args:
        vessel_data (dict): A dictionary containing vessel information.

    Returns:
        dict[str, float]: A dictionary with calculated tariff values.
                          The keys must exactly match: ['Port Dues'].
    """

    # Define rates as per calculation_rules for 'Port Dues'
    # Source: calculation_rules -> Port Dues -> rates
    BASIC_FEE_RATE_PER_100_TONS = 192.73  # ZAR per 100 tons
    INCREMENTAL_FEE_RATE_PER_100_TONS_PER_24H = 57.79  # ZAR per 100 tons per 24 hours

    calculated_tariffs = {}

    # --- Port Dues Calculation ---
    # Parameters needed:
    # - gross_tonnage: enriched_vessel_data.technical_specs.gross_tonnage
    # - days_alongside: enriched_vessel_data.operational_data.days_alongside

    # Accessing parameters from vessel_data
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    days_alongside = vessel_data.get("operational_data", {}).get("days_alongside")

    # Hard requirement: "If a needed parameter is missing, use a clearly named fallback only when the rules justify it."
    # The calculation rules for Port Dues do not specify fallbacks for 'gross_tonnage' or 'days_alongside'.
    # These are fundamental to the calculation. If they are missing or invalid, the calculation cannot proceed.
    # Therefore, a ValueError is raised to indicate missing essential data.
    if not isinstance(gross_tonnage, (int, float)) or gross_tonnage <= 0:
        raise ValueError("Missing or invalid 'gross_tonnage' in vessel_data. Cannot calculate Port Dues. Gross tonnage must be a positive number.")
    if not isinstance(days_alongside, (int, float)) or days_alongside < 0:
        # days_alongside can be 0, but not negative.
        raise ValueError("Missing or invalid 'days_alongside' in vessel_data. Cannot calculate Port Dues. Days alongside must be a non-negative number.")

    # Apply the formula as per calculation_rules -> Port Dues -> formula:
    # Port Dues = (Basic_Fee_Rate_per_100_tons * ceil(gross_tonnage / 100)) +
    #             (Incremental_Fee_Rate_per_100_tons_per_24h * ceil(gross_tonnage / 100) * days_alongside)

    # Rule: "A part of a 100 tons is rounded up to the next 100 tons (using ceil function)."
    tonnage_units = math.ceil(gross_tonnage / 100)

    # Calculate Basic Port Dues Fee
    # Rule: "A basic fee is charged per 100 tons or part thereof of the vessel's gross tonnage."
    basic_port_dues_fee = BASIC_FEE_RATE_PER_100_TONS * tonnage_units

    # Calculate Time-based Incremental Port Dues Fee
    # Rule: "An additional fee is charged per 100 tons or part thereof for each 24-hour period,
    #        with a part of a 24-hour period being applied pro rata."
    # "pro rata" means the `days_alongside` value (which can be a float) is used directly.
    incremental_port_dues_fee = INCREMENTAL_FEE_RATE_PER_100_TONS_PER_24H * tonnage_units * days_alongside

    # Total Port Dues
    total_port_dues = basic_port_dues_fee + incremental_port_dues_fee

    # Rule: "Final currency amounts are typically rounded to two decimal places."
    calculated_tariffs['Port Dues'] = round(total_port_dues, 2)

    return calculated_tariffs
