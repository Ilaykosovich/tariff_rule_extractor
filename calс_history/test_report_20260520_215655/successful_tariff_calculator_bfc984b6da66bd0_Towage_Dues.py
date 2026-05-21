import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """Calculate Towage Dues for a vessel."""
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    port = vessel_data.get('port')
    
    # Get time context from enriched data
    enriched_data = vessel_data.get('enriched_vessel_data', vessel_data)
    derived_context = enriched_data.get('derived_context', {})
    date_time_context = derived_context.get('date_time_context', {})
    
    arrival_context = date_time_context.get('operational_data.arrival_time', {})
    departure_context = date_time_context.get('operational_data.departure_time', {})
    
    arrival_standard_hours = arrival_context.get('is_standard_working_hours', True)
    departure_standard_hours = departure_context.get('is_standard_working_hours', True)
    arrival_weekend = arrival_context.get('is_weekend', False)
    departure_weekend = departure_context.get('is_weekend', False)
    
    # Fallbacks
    if gross_tonnage is None:
        gross_tonnage = 10000  # Default mid-range tonnage
    if port is None:
        port = "Durban"  # Default port
    
    # Durban rates by tonnage range
    durban_rates = [
        {"range": "Up to 2,000", "max_tonnage": 2000, "base_rate": 8140.0},
        {"range": "2,001 to 10,000", "max_tonnage": 10000, "base_rate": 12633.99, 
         "incremental_rate": 268.99, "incremental_threshold": 2000},
        {"range": "10,000 to 50,000", "max_tonnage": 50000, "base_rate": 38494.51, 
         "incremental_rate": 84.95, "incremental_threshold": 10000},
        {"range": "50,001 to 100,000", "max_tonnage": 100000, "base_rate": 73118.07, 
         "incremental_rate": 32.24, "incremental_threshold": 50000},
        {"range": "Above 100,000", "max_tonnage": float('inf'), "base_rate": 93548.13, 
         "incremental_rate": 23.65, "incremental_threshold": 100000}
    ]
    
    # Find applicable rate structure
    rate_structure = None
    for rate in durban_rates:
        if gross_tonnage <= rate["max_tonnage"]:
            rate_structure = rate
            break
    
    if rate_structure is None:
        rate_structure = durban_rates[-1]  # Use highest range as fallback
    
    # Calculate base fee
    base_fee = rate_structure["base_rate"]
    
    # Calculate incremental charges if applicable
    incremental_charge = 0.0
    if "incremental_rate" in rate_structure and "incremental_threshold" in rate_structure:
        if gross_tonnage > rate_structure["incremental_threshold"]:
            excess_tonnage = gross_tonnage - rate_structure["incremental_threshold"]
            # Per 100 tons or part thereof
            incremental_units = math.ceil(excess_tonnage / 100)
            incremental_charge = incremental_units * rate_structure["incremental_rate"]
    
    # Calculate total base towage fee (assuming both arrival and departure services)
    total_base_fee = (base_fee + incremental_charge) * 2
    
    # Apply surcharges for non-standard working hours or weekends
    surcharge_multiplier = 1.0
    
    # Check if either arrival or departure requires surcharge
    arrival_surcharge = (not arrival_standard_hours) or arrival_weekend
    departure_surcharge = (not departure_standard_hours) or departure_weekend
    
    if arrival_surcharge or departure_surcharge:
        # 25% surcharge applies
        surcharge_multiplier = 1.25
    
    total_towage_dues = total_base_fee * surcharge_multiplier
    
    return {
        'Towage Dues': total_towage_dues
    }
