import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Light Dues for a vessel calling at a South African port.
    
    Light dues are calculated based on:
    - Self-propelled vessels at registered port: per metre LOA per financial year
    - All other vessels: per 100 tons or part thereof based on gross tonnage
    """
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    loa_meters = vessel_data.get('technical_specs', {}).get('loa_meters')
    flag = vessel_data.get('vessel_metadata', {}).get('flag', '')
    port = vessel_data.get('port', '')
    
    # Get port stay duration from derived context
    port_stay_days = vessel_data.get('derived_context', {}).get('operational_characteristics', {}).get('port_stay_duration_days', 0)
    
    # Fallback values if missing
    if gross_tonnage is None:
        gross_tonnage = 0
    if loa_meters is None:
        loa_meters = 0
    
    # Determine if vessel is "self-propelled vessel at registered port"
    # Based on the rules, this applies to vessels licensed by Department of Environmental Affairs and Tourism at registered port
    # Malta flagged vessel calling at Durban would not qualify as "at registered port"
    is_registered_port_vessel = False
    if 'South Africa' in flag or 'RSA' in flag or 'ZA' in flag:
        # Only South African flagged vessels could potentially be "at registered port"
        # Additional licensing requirements would need to be verified but not available in data
        is_registered_port_vessel = False  # Conservative approach without licensing info
    
    # Check if vessel stays >60 days (triggers monthly coastal charging)
    is_extended_stay = port_stay_days > 60
    
    # Calculate light dues
    if is_registered_port_vessel:
        # Self-propelled vessels at registered port: 24.64 per metre LOA per financial year
        light_dues = loa_meters * 24.64
    else:
        # All other vessels: 117.08 per 100 tons or part thereof
        # Round up to next whole number of 100-ton increments
        tonnage_increments = math.ceil(gross_tonnage / 100)
        light_dues = tonnage_increments * 117.08
        
        # If extended stay >60 days, vessel is deemed coastal and charged monthly
        # However, the rules don't specify the exact monthly rate calculation
        # The base calculation remains the same as it's charged at first SA port
        if is_extended_stay:
            # Monthly basis would apply but specific monthly rate not provided
            # Base calculation applies as charged at first port of call
            pass
    
    return {
        'Light Dues': light_dues
    }
