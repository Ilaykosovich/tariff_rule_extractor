def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculates the 'Running Lines' tariff based on the provided vessel data and rules.

    This module implements the calculation for 'Running Lines' as per the
    specified calculation rules and rates. It adheres to the hard requirements
    of using only the standard library, defining a specific function signature,
    and handling missing parameters with justified fallbacks.

    Args:
        vessel_data (dict): A dictionary containing vessel information, including
                            technical specifications, operational data, and derived context.
                            Expected structure includes:
                            - 'derived_context.date_time_context.operational_data.arrival_time.is_standard_working_hours'
                            - 'derived_context.date_time_context.operational_data.departure_time.is_standard_working_hours'
                            - 'port' (implicitly used for rate selection, assumed 'Durban' -> 'Other Ports')

    Returns:
        dict[str, float]: A dictionary with the calculated 'Running Lines' tariff.
                          The key will be 'Running Lines'.
    """

    # --- Define rates and constants from calculation_rules ---
    # Rates for 'Other Ports' (Durban falls under 'Other Ports' as per rules)
    # Rule: "Per service rate for 'Other Ports'"
    BASE_SERVICE_RATE_OTHER_PORTS = 1654.56

    # Rule: "Minimum rate per service if service terminates or commences outside ordinary working hours for 'Other Ports'"
    OWH_MINIMUM_RATE_OTHER_PORTS = 3309.05

    # Rule: "Value Added Tax (VAT) rate"
    VAT_RATE = 0.15  # 15%

    # Rule: "Berthing and unberthing are charged as two separate services."
    NUMBER_OF_SERVICES = 2

    # --- Running Lines Calculation ---
    running_lines_cost = 0.0

    # Determine if any service (arrival or departure) is outside ordinary working hours (OWH).
    # Rule: "If the service terminates or commences outside ordinary working hours, a higher minimum charge applies per service."
    # Ambiguity handling: We use arrival/departure times as proxies for service commencement/termination.
    # Fallback strategy: If 'is_standard_working_hours' is missing, we assume True (standard working hours)
    # as per 'assumptions': "Use standard working time when the tariff text does not define special working hours."
    # and "Default standard working time is Monday-Friday 08:00-17:00 local port time."

    # Safely get the 'is_standard_working_hours' flag for arrival
    arrival_is_standard_working_hours = vessel_data.get('derived_context', {}) \
                                        .get('date_time_context', {}) \
                                        .get('operational_data.arrival_time', {}) \
                                        .get('is_standard_working_hours', True)

    # Safely get the 'is_standard_working_hours' flag for departure
    departure_is_standard_working_hours = vessel_data.get('derived_context', {}) \
                                         .get('date_time_context', {}) \
                                         .get('operational_data.departure_time', {}) \
                                         .get('is_standard_working_hours', True)

    # Check if either arrival or departure is outside standard working hours
    is_any_service_owh = not arrival_is_standard_working_hours or not departure_is_standard_working_hours

    # Calculate the cost per service based on OWH status
    cost_per_service = BASE_SERVICE_RATE_OTHER_PORTS
    if is_any_service_owh:
        cost_per_service = OWH_MINIMUM_RATE_OTHER_PORTS

    # Calculate the subtotal for all services before VAT
    subtotal_before_vat = cost_per_service * NUMBER_OF_SERVICES

    # Apply VAT (Rule: "All tariffs are subject to VAT at 15%.")
    running_lines_cost = subtotal_before_vat * (1 + VAT_RATE)

    # Round the final cost to two decimal places for currency
    running_lines_cost = round(running_lines_cost, 2)

    # --- Surcharges and Additional Fees (Not applicable based on current data and rules interpretation) ---
    # - Delay Surcharge: The vessel_data does not contain "notified time" or "service completed time"
    #   required to calculate delays as per the rule.
    # - Cancellation Fee for OWH Standby: The vessel_data does not contain information about
    #   cancellations or standby periods.
    # - Damage to Ropes: This is an event-based charge not indicated in the provided data.
    # - Gross Tonnage, Vessel Type, Days Alongside: These parameters were considered but not used
    #   as the explicit 'Running Lines' tariff (Section 3.9) does not specify charges based on them,
    #   and ambiguities regarding other sections (e.g., 3.8 'BERTHING SERVICES') were resolved
    #   by prioritizing Section 3.9.

    return {
        'Running Lines': running_lines_cost
    }
