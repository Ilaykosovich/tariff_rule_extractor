def calculate(vessel_data: dict) -> dict[str, float]:
    """Calculate Towage Dues for vessel operations."""
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    port = vessel_data.get('port')
    
    # Get time context for surcharge determination
    derived_context = vessel_data.get('derived_context', {})
    date_time_context = derived_context.get('date_time_context', {})
    
    arrival_context = date_time_context.get('operational_data.arrival_time', {})
    departure_context = date_time_context.get('operational_data.departure_time', {})
    
    arrival_standard_hours = arrival_context.get('is_standard_working_hours', True)
    departure_standard_hours = departure_context.get('is_standard_working_hours', True)
    arrival_weekend = arrival_context.get('is_weekend', False)
    departure_weekend = departure_context.get('is_weekend', False)
    
    # Fallbacks for missing parameters
    if gross_tonnage is None:
        gross_tonnage = 10000  # Default to mid-range tonnage
    if port is None:
        port = "Durban"  # Default port
    
    # Durban rates by tonnage range
    durban_rates = [
        {"min": 0, "max": 2000, "base": 8140.0, "incremental": None, "threshold": None},
        {"min": 2001, "max": 10000, "base": 12633.99, "incremental": 268.99, "threshold": 2000},
        {"min": 10001, "max": 50000, "base": 38494.51, "incremental": 84.95, "threshold": 10000},
        {"min": 50001, "max": 100000, "base": 73118.07, "incremental": 32.24, "threshold": 50000},
        {"min": 100001, "max": float('inf'), "base": 93548.13, "incremental": 23.65, "threshold": 100000}
    ]
    
    # Find applicable rate tier
    rate_tier = None
    for tier in durban_rates:
        if tier["min"] <= gross_tonnage <= tier["max"]:
            rate_tier = tier
            break
    
    if rate_tier is None:
        # Fallback to highest tier
        rate_tier = durban_rates[-1]
    
    # Calculate base fee
    base_fee = rate_tier["base"]
    
    # Calculate incremental charges if applicable
    incremental_charge = 0.0
    if rate_tier["incremental"] is not None and rate_tier["threshold"] is not None:
        excess_tonnage = gross_tonnage - rate_tier["threshold"]
        if excess_tonnage > 0:
            # Per 100 tons or part thereof
            incremental_units = int((excess_tonnage + 99) // 100)  # Ceiling division
            incremental_charge = incremental_units * rate_tier["incremental"]
    
    # Calculate total base towage fee (assuming both arrival and departure services)
    total_base_fee = (base_fee + incremental_charge) * 2
    
    # Apply surcharges for non-standard working hours or weekends
    surcharge_multiplier = 1.0
    
    # Check arrival service
    if not arrival_standard_hours or arrival_weekend:
        surcharge_multiplier += 0.25
    
    # Check departure service  
    if not departure_standard_hours or departure_weekend:
        surcharge_multiplier += 0.25
    
    # Calculate final towage dues
    towage_dues = total_base_fee * surcharge_multiplier
    
    return {
        'Towage Dues': towage_dues
    }
