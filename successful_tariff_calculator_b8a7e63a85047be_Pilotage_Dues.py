import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Pilotage Dues for vessels at South African ports.
    
    Args:
        vessel_data: Dictionary containing vessel and operational data
        
    Returns:
        Dictionary with calculated tariff values
    """
    
    # Extract required data
    port = vessel_data.get('port', '')
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage', 0)
    num_operations = vessel_data.get('operational_data', {}).get('num_operations', 1)
    
    # Get working hours status from derived context
    derived_context = vessel_data.get('derived_context', {})
    date_time_context = derived_context.get('date_time_context', {})
    
    arrival_working_hours = date_time_context.get('operational_data.arrival_time', {}).get('is_standard_working_hours', True)
    departure_working_hours = date_time_context.get('operational_data.departure_time', {}).get('is_standard_working_hours', True)
    
    # Port-specific rates for Durban
    if port == 'Durban':
        basic_fee_per_service = 18608.61
        rate_per_100_tons = 9.72
    else:
        # Fallback for other ports (using Durban rates as default)
        basic_fee_per_service = 18608.61
        rate_per_100_tons = 9.72
    
    # Calculate tonnage component
    # Round up to next 100 tons (part thereof)
    tonnage_units = math.ceil(gross_tonnage / 100)
    tonnage_charge_per_service = tonnage_units * rate_per_100_tons
    
    # Base charge per service
    base_charge_per_service = basic_fee_per_service + tonnage_charge_per_service
    
    # Calculate total for all operations
    total_base_charge = base_charge_per_service * num_operations
    
    # Apply surcharges for services outside ordinary working hours
    surcharge_multiplier = 1.0
    
    # Check if any pilotage service occurs outside working hours
    # Assuming arrival requires inbound pilotage and departure requires outbound pilotage
    services_outside_hours = 0
    
    if not arrival_working_hours:
        services_outside_hours += 1
    if not departure_working_hours:
        services_outside_hours += 1
    
    # Calculate surcharge (50% for services outside working hours)
    if services_outside_hours > 0:
        surcharge_amount = (services_outside_hours * base_charge_per_service) * 0.5
        total_pilotage_dues = total_base_charge + surcharge_amount
    else:
        total_pilotage_dues = total_base_charge
    
    return {
        'Pilotage Dues': total_pilotage_dues
    }
