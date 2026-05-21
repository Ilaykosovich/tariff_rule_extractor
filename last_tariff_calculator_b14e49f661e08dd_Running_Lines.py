def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the 'Running Lines' tariff based on vessel data and predefined rules.

    This function expects the 'vessel_data' dictionary to potentially include
    'derived_context' information, as indicated by the 'selected_parameters'
    in the problem description (e.g., 'derived_context.date_time_context...').

    Args:
        vessel_data (dict): A dictionary containing vessel and operational data,
                            potentially enriched with derived context.

    Returns:
        dict[str, float]: A dictionary with the calculated tariff values.
                          The keys will exactly match the requested tariffs.
    """

    # --- 1. Define Rates (hardcoded as per instructions, not reading files) ---
    # These rates are directly extracted from the 'calculation_rules' provided.
    RATES = {
        "base_service": {
            "Port Elizabeth / Ngqura": 2266.73,
            "Cape Town": 2370.84,
            "Saldanha": 2085.59,
            "Other Ports": 1654.56,
        },
        "owh_minimum": {
            "Port Elizabeth / Ngqura": 4533.42,
            "Cape Town": 3309.05,  # Using 3309.05 as per the first table, acknowledging ambiguity.
            "Saldanha": 4171.18,
            "Other Ports": 3309.05,
        },
        "cancellation_owh_tug_vessel": 1654.56,  # R per hour
        "saldanha_remooring_without_tug_pilots": 4739.14,  # R per service
    }

    # --- 2. Extract necessary data with fallbacks ---
    # Access port from 'enriched_vessel_data' if present, otherwise from top-level 'vessel_data'.
    # This handles the structure where 'vessel_data' might be the 'enriched_input_context'.
    port = vessel_data.get("enriched_vessel_data", vessel_data).get("port")

    # Fallback for num_operations: Assume 2 services (berthing and unberthing) if not provided.
    # The rules state "Berthing and unberthing are typically considered separate services."
    # and the example has num_operations: 2.
    num_operations = vessel_data.get("enriched_vessel_data", vessel_data).get(
        "operational_data", {}
    ).get("num_operations", 2)
    if not isinstance(num_operations, (int, float)) or num_operations < 0:
        num_operations = 2  # Ensure it's a valid positive number

    # Get OWH status for arrival and departure from 'derived_context'.
    # If data is missing, assume standard working hours (not OWH) as per general assumptions.
    # 'is_standard_working_hours = True' means NOT OWH.
    # 'is_standard_working_hours = False' means OWH.
    # We need to check if the service is OWH.
    derived_context = vessel_data.get("derived_context", {})
    
    is_arrival_owh = not derived_context.get("date_time_context", {}).get(
        "operational_data.arrival_time", {}
    ).get("is_standard_working_hours", True)  # Default to True (standard hours) if missing

    is_departure_owh = not derived_context.get("date_time_context", {}).get(
        "operational_data.departure_time", {}
    ).get("is_standard_working_hours", True)  # Default to True (standard hours) if missing

    # --- 3. Determine Port-Specific Rate Category ---
    port_category = "Other Ports"
    if port in ["Port Elizabeth", "Ngqura"]:
        port_category = "Port Elizabeth / Ngqura"
    elif port == "Cape Town":
        port_category = "Cape Town"
    elif port == "Saldanha":
        port_category = "Saldanha"

    # --- 4. Retrieve Applicable Rates for the Port ---
    base_service_rate = RATES["base_service"].get(port_category, RATES["base_service"]["Other Ports"])
    owh_minimum_rate = RATES["owh_minimum"].get(port_category, RATES["owh_minimum"]["Other Ports"])

    # --- 5. Calculate Running Lines Charge ---
    total_running_lines_charge = 0.0
    services_calculated = 0

    # Rule: "Services for entering, leaving, or shifting a vessel are charged on a per-service basis.
    # Berthing and unberthing are typically considered separate services."
    # Rule: "For each service, the charge is the maximum of the 'base_service' rate and the 'owh_minimum' rate for the respective port."

    # Handle Berthing (Arrival) service
    if num_operations >= 1:
        charge_for_berthing = base_service_rate
        if is_arrival_owh:
            charge_for_berthing = max(base_service_rate, owh_minimum_rate)
        total_running_lines_charge += charge_for_berthing
        services_calculated += 1

    # Handle Unberthing (Departure) service
    if num_operations >= 2:
        charge_for_unberthing = base_service_rate
        if is_departure_owh:
            charge_for_unberthing = max(base_service_rate, owh_minimum_rate)
        total_running_lines_charge += charge_for_unberthing
        services_calculated += 1

    # Handle additional services (e.g., shifting) if num_operations > 2.
    # The rules do not provide OWH status for these additional services.
    # Fallback: Assume additional services are charged at the base rate, as we lack specific OWH context for them.
    if num_operations > services_calculated:
        additional_services_count = num_operations - services_calculated
        total_running_lines_charge += additional_services_count * base_service_rate

    # --- 6. Check for Port-Specific Rule (Saldanha Remooring without Tug/Pilots) ---
    # This rule is an exception: "A specific rate applies for remooring operations... superseding general service rates for this specific activity at this port."
    # `applies_if`: "Vessel is at the Port of Saldanha and performs remooring without tug/pilot assistance."
    # The provided vessel data has `port: "Durban"` and `activity: "Exporting Iron Ore"`.
    # This rule does not apply under these conditions. If it did, it would override the general service rates
    # for the specific remooring activity at Saldanha.

    # --- 7. Check for Delay Charges ---
    # `applies_if`: "Vessel arrival or departure is delayed by 30 minutes or more after the notified time."
    # The `vessel_data` does not contain "notified time" or "service completion time" to calculate delays.
    # Therefore, this rule cannot be applied. Assume no delays.

    # --- 8. Check for Cancellation Fee ---
    # `applies_if`: "Request for OWH tug/vessel service is cancelled after standby commenced."
    # The `vessel_data` does not contain information about cancellations or standby hours.
    # Therefore, this rule cannot be applied. Assume no cancellations.

    # --- 9. Rounding ---
    # "Not explicitly specified, typically 2 decimal places for currency (South African Rand)."
    total_running_lines_charge = round(total_running_lines_charge, 2)

    return {"Running Lines": total_running_lines_charge}
