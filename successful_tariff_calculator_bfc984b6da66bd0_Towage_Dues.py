import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """Calculate Towage Dues for a vessel at Durban port."""
    
    # Extract required data
    port = vessel_data.get("port", "")
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage")
    num_operations = vessel_data.get("operational_data", {}).get("num_operations", 1)
    
    # Get time context for surcharge determination
    enriched_data = vessel_data.get("enriched_vessel_data", {})
    derived_context = vessel_data.get("derived_context", {})
    date_time_context = derived_context.get("date_time_context", {})
    
    arrival_context = date_time_context.get("operational_data.arrival_time", {})
    departure_context = date_time_context.get("operational_data.departure_time", {})
    
    arrival_standard_hours = arrival_context.get("is_standard_working_hours", True)
    departure_standard_hours = departure_context.get("is_standard_working_hours", True)
    arrival_weekend = arrival_context.get("is_weekend", False)
    departure_weekend = departure_context.get("is_weekend", False)
    
    # Use gross tonnage as primary tonnage basis
    if gross_tonnage is None:
        gross_tonnage = 0
    
    # Durban port rates
    durban_rates = [
        {"range": "Up to 2,000", "min": 0, "max": 2000, "base": 8140.0, "incremental": 0, "threshold": 0},
        {"range": "2,001 to 10,000", "min": 2001, "max": 10000, "base": 12633.99, "incremental": 268.99, "threshold": 2000},
        {"range": "10,000 to 50,000", "min": 10001, "max": 50000, "base": 38494.51, "incremental": 84.95, "threshold": 10000},
        {"range": "50,001 to 100,000", "min": 50001, "max": 100000, "base": 73118.07, "incremental": 32.24, "threshold": 50000},
        {"range": "Above 100,000", "min": 100001, "max": float('inf'), "base": 93548.13, "incremental": 23.65, "threshold": 100000}
    ]
    
    # Find applicable rate tier
    applicable_rate = None
    for rate in durban_rates:
        if rate["min"] <= gross_tonnage <= rate["max"]:
            applicable_rate = rate
            break
    
    if applicable_rate is None:
        # Default to first tier if no match
        applicable_rate = durban_rates[0]
    
    # Calculate base fee
    base_fee = applicable_rate["base"]
    
    # Calculate incremental charges if applicable
    incremental_charge = 0
    if applicable_rate["incremental"] > 0 and gross_tonnage > applicable_rate["threshold"]:
        excess_tonnage = gross_tonnage - applicable_rate["threshold"]
        # Per 100 tons or part thereof
        incremental_units = math.ceil(excess_tonnage / 100)
        incremental_charge = incremental_units * applicable_rate["incremental"]
    
    # Calculate total base cost per service
    service_cost = base_fee + incremental_charge
    
    # Apply surcharges
    surcharge_multiplier = 1.0
    
    # 25% surcharge for services outside ordinary working hours or on weekends
    needs_surcharge = False
    if not arrival_standard_hours or not departure_standard_hours:
        needs_surcharge = True
    if arrival_weekend or departure_weekend:
        needs_surcharge = True
    
    if needs_surcharge:
        surcharge_multiplier = 1.25
    
    # Apply surcharge to service cost
    service_cost_with_surcharge = service_cost * surcharge_multiplier
    
    # Multiply by number of operations (each operation requires towage service)
    total_towage_dues = service_cost_with_surcharge * num_operations
    
    return {"Towage Dues": total_towage_dues}
