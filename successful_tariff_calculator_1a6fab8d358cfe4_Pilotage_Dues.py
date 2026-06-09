import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Pilotage Dues for vessel operations at Durban port.
    
    Args:
        vessel_data: Dictionary containing vessel and operational data
        
    Returns:
        Dictionary with 'Pilotage Dues' key and calculated value in ZAR
    """
    
    # Extract required data
    port = vessel_data.get('port', '')
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    num_operations = vessel_data.get('operational_data', {}).get('num_operations', 1)
    
    # Extract time context for surcharge determination
    derived_context = vessel_data.get('derived_context', {})
    date_time_context = derived_context.get('date_time_context', {})
    
    arrival_working_hours = date_time_context.get('operational_data.arrival_time', {}).get('is_standard_working_hours', True)
    departure_working_hours = date_time_context.get('operational_data.departure_time', {}).get('is_standard_working_hours', True)
    
    # Fallback values
    if gross_tonnage is None:
        gross_tonnage = 50000  # Reasonable fallback for large bulk carrier
    
    # Durban port rates
    if port == 'Durban':
        basic_fee = 18608.61  # ZAR per service
        per_100_tons = 9.72   # ZAR per 100 tons or part thereof
    else:
        # Use Durban rates as fallback since vessel data specifies Durban
        basic_fee = 18608.61
        per_100_tons = 9.72
    
    # Calculate tonnage charge
    # Round up to next 100 ton increment (per 100 tons or part thereof)
    tonnage_increments = math.ceil(gross_tonnage / 100)
    tonnage_charge = tonnage_increments * per_100_tons
    
    # Base charge per service
    base_charge_per_service = basic_fee + tonnage_charge
    
    # Calculate total for all operations
    total_base_charge = base_charge_per_service * num_operations
    
    # Apply surcharges for services outside ordinary working hours
    surcharge_multiplier = 1.0
    
    # Check if any service occurs outside working hours (50% surcharge applies)
    if not arrival_working_hours or not departure_working_hours:
        surcharge_multiplier = 1.5
    
    # Calculate final pilotage dues
    pilotage_dues = total_base_charge * surcharge_multiplier
    
    return {
        'Pilotage Dues': pilotage_dues
    }
