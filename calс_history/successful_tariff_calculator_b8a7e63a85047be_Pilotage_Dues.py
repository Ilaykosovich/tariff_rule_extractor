import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """Calculate Pilotage Dues for vessel operations."""
    
    # Extract required data
    port = vessel_data.get('port', '')
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    num_operations = vessel_data.get('operational_data', {}).get('num_operations', 1)
    
    # Extract working hours information from enriched context
    enriched_data = vessel_data.get('enriched_vessel_data', vessel_data)
    derived_context = enriched_data.get('derived_context', {})
    date_time_context = derived_context.get('date_time_context', {})
    
    arrival_working_hours = date_time_context.get('operational_data.arrival_time', {}).get('is_standard_working_hours', True)
    departure_working_hours = date_time_context.get('operational_data.departure_time', {}).get('is_standard_working_hours', True)
    
    # Fallback for missing gross tonnage
    if gross_tonnage is None:
        gross_tonnage = 0
    
    # Port-specific rates for Durban
    if port == 'Durban':
        basic_fee = 18608.61
        per_100_tons = 9.72
    else:
        # Fallback rates (using Durban as default since it's the only port with specified rates)
        basic_fee = 18608.61
        per_100_tons = 9.72
    
    # Calculate tonnage charge (rounded up to next 100 tons)
    tonnage_units = math.ceil(gross_tonnage / 100)
    tonnage_charge = tonnage_units * per_100_tons
    
    # Base pilotage cost per service
    base_cost_per_service = basic_fee + tonnage_charge
    
    # Calculate total cost based on number of operations
    # Assuming each operation requires pilotage service
    total_base_cost = base_cost_per_service * num_operations
    
    # Apply surcharges for non-working hours
    total_surcharge = 0
    
    # Check arrival surcharge (if arrival is outside working hours)
    if not arrival_working_hours:
        total_surcharge += base_cost_per_service * 0.5
    
    # Check departure surcharge (if departure is outside working hours and there are 2 operations)
    if num_operations >= 2 and not departure_working_hours:
        total_surcharge += base_cost_per_service * 0.5
    
    total_pilotage_dues = total_base_cost + total_surcharge
    
    return {
        'Pilotage Dues': total_pilotage_dues
    }
