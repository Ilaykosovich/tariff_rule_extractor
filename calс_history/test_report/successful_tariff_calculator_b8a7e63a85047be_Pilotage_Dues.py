import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """Calculate Pilotage Dues for vessel operations."""
    
    # Extract required data
    port = vessel_data.get("port", "")
    gross_tonnage = vessel_data.get("technical_specs", {}).get("gross_tonnage", 0)
    num_operations = vessel_data.get("operational_data", {}).get("num_operations", 2)  # Default 2 for arrival/departure
    
    # Get working hours status from enriched context
    enriched_data = vessel_data.get("enriched_vessel_data", vessel_data)
    derived_context = enriched_data.get("derived_context", {})
    date_time_context = derived_context.get("date_time_context", {})
    
    arrival_working_hours = date_time_context.get("operational_data.arrival_time", {}).get("is_standard_working_hours", True)
    departure_working_hours = date_time_context.get("operational_data.departure_time", {}).get("is_standard_working_hours", True)
    
    # Durban rates
    if port == "Durban":
        basic_fee_per_service = 18608.61
        rate_per_100_tons = 9.72
    else:
        # Fallback for other ports (using Durban rates as default)
        basic_fee_per_service = 18608.61
        rate_per_100_tons = 9.72
    
    # Calculate tonnage charge (per 100 tons or part thereof - round up)
    tonnage_units = math.ceil(gross_tonnage / 100)
    tonnage_charge_per_service = tonnage_units * rate_per_100_tons
    
    # Base charge per service
    base_charge_per_service = basic_fee_per_service + tonnage_charge_per_service
    
    # Calculate total for all operations
    total_pilotage = 0
    
    # Apply charges for each operation
    for i in range(num_operations):
        service_charge = base_charge_per_service
        
        # Apply 50% surcharge if outside ordinary working hours
        # Assume first operation is arrival, second is departure
        if i == 0 and not arrival_working_hours:  # Arrival outside working hours
            service_charge *= 1.5
        elif i == 1 and not departure_working_hours:  # Departure outside working hours
            service_charge *= 1.5
        
        total_pilotage += service_charge
    
    return {
        "Pilotage Dues": total_pilotage
    }
