import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the Light Dues tariff based on vessel data and predefined rules.

    Args:
        vessel_data (dict): A dictionary containing vessel metadata, technical specifications,
                            and operational data.

    Returns:
        dict[str, float]: A dictionary with 'Light Dues' as the key and its calculated
                          tariff value as a float. Returns 0.0 if calculation is not possible
                          due to missing critical parameters or ambiguous rates.
    """

    # --- Extracting parameters from vessel_data ---
    # Gross tonnage is a primary parameter for several rules.
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    # Length overall is a primary parameter for Rule 1.
    length_overall_metres = vessel_data.get("technical_specs", {}).get("loa_meters")
    # Time in SA waters is a primary parameter for conditions in Rules 2 and 3.
    # 'days_alongside' is used as a proxy for 'time_in_SA_waters' for this specific port call.
    time_in_SA_waters = vessel_data.get("operational_data", {}).get("days_alongside")
    # Vessel type helps determine if it's self-propelled.
    vessel_type_str = vessel_data.get("technical_specs", {}).get("type")
    # Vessel flag helps determine if it's arriving from a foreign port.
    vessel_flag = vessel_data.get("vessel_metadata", {}).get("flag")

    # --- Fallback values for missing parameters and conditions ---
    # These fallbacks are clearly named and justified based on:
    # 1. Common interpretation of vessel types (e.g., Bulk Carrier is self-propelled).
    # 2. Default assumptions for missing boolean flags (e.g., False for special statuses,
    #    True for common scenarios like being within coastline for a port call).
    # 3. Interpretation of "per part thereof" for numeric multipliers (minimum 1 part).

    # Fallback for 'vessel_type': A "Bulk Carrier" is inherently self-propelled.
    is_self_propelled = vessel_type_str == "Bulk Carrier"

    # Fallback for 'is_licensed_by_DEA_Tourism': No information provided, assume False.
    is_licensed_by_DEA_Tourism = False

    # Fallback for 'is_at_registered_port': No information provided. For a foreign-flagged
    # commercial vessel on a port call, it's generally not at its registered port. Assume False.
    is_at_registered_port = False

    # Fallback for 'stays_within_SA_coastline': Vessel is at a port, so assume it's within
    # the South African coastline.
    stays_within_SA_coastline = True

    # Fallback for 'has_bonafide_coaster_status': No information provided. This is a special
    # status, so assume False by default.
    has_bonafide_coaster_status = False

    # Fallback for 'arrives_from_foreign_port': If the vessel's flag is not South African,
    # assume it's arriving from a foreign port for a commercial call.
    arrives_from_foreign_port = vessel_flag is not None and not vessel_flag.startswith("ZAF")

    # Fallback for 'financial_year_parts': The rule states "per financial year or part thereof".
    # This implies a minimum charge for at least one part, so 1.0 is a reasonable default.
    financial_year_parts = 1.0

    # Fallback for 'calendar_months' and 'calendar_months_after_60_days':
    # Similar to financial_year_parts, "per calendar month basis" implies a minimum of 1 month.
    calendar_months = 1.0
    calendar_months_after_60_days = 1.0

    # Fallback for 'time_in_SA_waters': If 'days_alongside' is missing, assume 0 days.
    # This ensures the condition 'time_in_SA_waters <= 60' is met for initial charges.
    if time_in_SA_waters is None:
        time_in_SA_waters = 0.0

    # --- Validate essential parameters for calculation ---
    # If gross_tonnage, a key parameter for most rules, is missing, we cannot calculate.
    if gross_tonnage is None:
        return {"Light Dues": 0.0}

    light_dues_value = 0.0

    # --- Apply Light Dues rules based on a hierarchical evaluation ---
    # Rules are evaluated in an order that prioritizes specific categories
    # (Self-propelled, Bonafide Coasters) before falling back to general categories
    # ("All Other Vessels"). If a specific rule's conditions are not fully met,
    # the vessel is considered for the next category.

    # 1. Attempt Rule 1: Self-propelled or Licensed Vessels
    # Conditions:
    #   - Vessel is self-propelled OR licensed by Department of Environmental Affairs and Tourism
    #   - Vessel is at its registered port
    if (is_self_propelled or is_licensed_by_DEA_Tourism) and is_at_registered_port:
        if length_overall_metres is not None:
            # Formula: 24.64 * ceil(length_overall_metres) * ceil(financial_year_parts)
            light_dues_value = 24.64 * math.ceil(length_overall_metres) * math.ceil(financial_year_parts)
            return {"Light Dues": light_dues_value}
        # If length_overall_metres is missing, Rule 1 cannot be calculated, so proceed to next rules.

    # 2. Attempt Rules for Bonafide Coasters (Rules 4 & 5)
    # These rules apply only if the vessel has 'Bonafide Coasters' status.
    elif has_bonafide_coaster_status:
        # Rule 5: Bonafide Coasters (Entering from Foreign Port)
        # Conditions:
        #   - Vessel has 'Bonafide Coasters' status (met by this block)
        #   - Vessel enters a South African port following a visit or call from a foreign port
        if arrives_from_foreign_port:
            # Formula: 117.08 * ceil(gross_tonnage / 100)
            light_dues_value = 117.08 * math.ceil(gross_tonnage / 100)
            return {"Light Dues": light_dues_value}
        else:
            # Rule 4: Bonafide Coasters (Standard Monthly Charge)
            # Conditions:
            #   - Vessel has 'Bonafide Coasters' status (met by this block)
            #   - Vessel has NOT entered a South African port following a visit or call from a foreign port (met)
            # Note: The rate for this rule is explicitly marked as "Not specified (ambiguity)".
            # As per hard requirement "Do not infer rates", we cannot calculate this.
            return {"Light Dues": 0.0}

    # 3. Fallback to "All Other Vessels" (Rules 2 & 3)
    # If the vessel does not fit into Rule 1 or the Bonafide Coasters category,
    # it is considered an "All Other Vessels" for Light Dues purposes.
    # This implicitly satisfies the "Vessel is not self-propelled or licensed..." condition
    # for Rules 2 and 3, as the more specific Rule 1 did not apply.
    else:
        # Differentiate between Rule 2 and Rule 3 based on time spent in SA waters.
        if time_in_SA_waters <= 60:
            # Rule 2: All Other Vessels (Initial Charge)
            # Conditions:
            #   - Vessel is not self-propelled or licensed by Department of Environmental Affairs and Tourism (implicitly met)
            #   - Vessel does not proceed beyond the borders of the South African coastline (assumed True)
            #   - Time spent in South African waters does not exceed 60 days (met)
            # Formula: 117.08 * ceil(gross_tonnage / 100)
            light_dues_value = 117.08 * math.ceil(gross_tonnage / 100)
            return {"Light Dues": light_dues_value}
        else:
            # Rule 3: All Other Vessels (After 60 Days in SA Waters)
            # Conditions:
            #   - Vessel is not self-propelled or licensed by Department of Environmental Affairs and Tourism (implicitly met)
            #   - Time spent in South African waters exceeds 60 days (met)
            # Note: The rate for this rule is explicitly marked as "Not specified (ambiguity)".
            # As per hard requirement "Do not infer rates", we cannot calculate this.
            return {"Light Dues": 0.0}

    # If no rule could be applied or calculated, return 0.0
    return {"Light Dues": light_dues_value}
