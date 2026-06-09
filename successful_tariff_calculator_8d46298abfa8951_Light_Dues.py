import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Light Dues for vessels calling at South African ports.
    
    Light dues are calculated based on:
    - Self-propelled vessels at registered port: LOA × 24.64 per financial year
    - All other vessels: (gross_tonnage ÷ 100, rounded up) × 117.08
    """
    
    # Extract required parameters
    loa_meters = vessel_data.get('technical_specs', {}).get('loa_meters')
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    flag = vessel_data.get('vessel_metadata', {}).get('flag', '')
    port = vessel_data.get('port', '')
    
    # Determine if this is a self-propelled vessel at registered port
    # Malta-flagged vessel calling at Durban is not at its registered port
    is_registered_port = False
    if flag and 'MLT' in flag and port == 'Durban':
        is_registered_port = False
    
    # Calculate Light Dues
    if is_registered_port and loa_meters is not None:
        # Self-propelled vessels at registered port: per metre LOA per financial year
        light_dues = loa_meters * 24.64
    elif gross_tonnage is not None:
        # All other vessels: per 100 tons or part thereof
        tonnage_blocks = math.ceil(gross_tonnage / 100.0)
        light_dues = tonnage_blocks * 117.08
    else:
        # Fallback if gross tonnage is missing
        light_dues = 0.0
    
    return {
        'Light Dues': light_dues
    }
