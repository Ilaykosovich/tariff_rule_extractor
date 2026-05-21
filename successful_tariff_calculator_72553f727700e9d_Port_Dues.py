from decimal import Decimal, getcontext, ROUND_CEILING

# Set the precision for Decimal calculations. Default is usually 28, which is sufficient.
# getcontext().prec = 28

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the Port Dues for a vessel based on provided data and specific rules.

    Args:
        vessel_data (dict): A dictionary containing vessel information,
                            including technical specifications and operational data.
                            This dictionary is expected to have the structure
                            similar to the 'enriched_vessel_data' in the context.

    Returns:
        dict[str, float]: A dictionary with 'Port Dues' as the key and its
                          calculated value as a float.
    """
    # Define rates using Decimal for precise monetary calculations
    BASIC_FEE_RATE = Decimal('192.73')
    TIME_BASED_FEE_RATE = Decimal('57.79')

    # Extract required parameters from vessel_data.
    # Using .get() with default empty dictionaries provides robustness against missing keys
    # in intermediate levels. If the final parameter (gross_tonnage or days_alongside)
    # is missing, a ValueError is raised as the rules do not specify a fallback for these
    # core calculation inputs.
    gross_tonnage_raw = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    days_alongside_raw = vessel_data.get("operational_data", {}).get("days_alongside")

    if gross_tonnage_raw is None:
        raise ValueError("Missing 'gross_tonnage' in vessel_data.technical_specs. Cannot calculate Port Dues.")
    if days_alongside_raw is None:
        raise ValueError("Missing 'days_alongside' in vessel_data.operational_data. Cannot calculate Port Dues.")

    # Convert extracted values to Decimal for accurate calculations.
    # Converting via string is the safest way to avoid floating-point inaccuracies
    # when the input might originate from a float.
    gross_tonnage = Decimal(str(gross_tonnage_raw))
    days_alongside = Decimal(str(days_alongside_raw))

    # --- Port Dues Calculation Logic ---

    # Rule: "Gross Tonnage is rounded up to the nearest 100 tons for calculation."
    # This means we divide the gross tonnage by 100, and then round that result up
    # to the nearest whole number to get the number of "100-ton units or part thereof".
    # Example: 51300 GT -> 513 units; 51301 GT -> 514 units.
    gross_tonnage_units = (gross_tonnage / Decimal('100')).to_integral_value(rounding=ROUND_CEILING)

    # Rule: "A basic fee is charged per 100 tons or part thereof."
    basic_fee = gross_tonnage_units * BASIC_FEE_RATE

    # Rule: "An additional fee is charged per 100 tons or part thereof per 24 hour period,
    # with a part of a 24 hour period being applied pro rata."
    # The 'days_alongside' parameter directly represents the duration in 24-hour periods,
    # including fractional parts, which aligns with the "pro rata" application.
    time_based_fee = gross_tonnage_units * TIME_BASED_FEE_RATE * days_alongside

    # Total Port Dues is the sum of the basic fee and the time-based fee.
    port_dues_decimal = basic_fee + time_based_fee

    # --- Application of Reductions/Surcharges ---
    # Based on the provided 'calculation_rules' and their 'applies_to_vessel: false'
    # and 'reason_not_applicable' fields, none of the listed reductions or surcharges
    # (e.g., for non-cargo working, bunker calls, short stay, long stay, or specific vessel types)
    # are applicable to this vessel and scenario.
    # Therefore, no further adjustments are made to the calculated port_dues_decimal.

    # The function is required to return a dictionary with float values.
    return {
        'Port Dues': float(port_dues_decimal)
    }
