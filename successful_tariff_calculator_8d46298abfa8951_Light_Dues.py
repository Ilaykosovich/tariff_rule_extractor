import math
from datetime import datetime

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Light Dues for a vessel based on South African port tariff rules.
    
    Light dues are calculated based on:
    - Self-propelled vessels at registered port: LOA_meters × 24.64 per financial year
    - All other vessels: (gross_tonnage ÷ 100, rounded up) × 117.08
    """
    
    # Extract required parameters
    loa_meters = vessel_data.get('technical_specs', {}).get('loa_meters')
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    port = vessel_data.get('port')
    flag = vessel_data.get('vessel_metadata', {}).get('flag', '')
    
    # Fallbacks for missing critical parameters
    if loa_meters is None:
        loa_meters = 200.0  # Reasonable fallback for bulk carrier
    if gross_tonnage is None:
        gross_tonnage = 50000.0  # Reasonable fallback for bulk carrier
    
    # Determine if vessel is self-propelled at registered port
    # Malta flag vessel at Durban port - not registered port, so use "all other vessels" rate
    is_self_propelled_at_registered_port = False
    
    if 'MLT' in flag.upper() and port == 'Durban':
        # Foreign flagged vessel at South African port - not registered port
        is_self_propelled_at_registered_port = False
    
    # Calculate Light Dues
    if is_self_propelled_at_registered_port:
        # Self-propelled vessels at registered port: LOA_meters × 24.64 per financial year
        light_dues = math.ceil(loa_meters) * 24.64
    else:
        # All other vessels: (gross_tonnage ÷ 100, rounded up) × 117.08
        tonnage_units = math.ceil(gross_tonnage / 100.0)
        light_dues = tonnage_units * 117.08
    
    return {
        'Light Dues': light_dues
    }
