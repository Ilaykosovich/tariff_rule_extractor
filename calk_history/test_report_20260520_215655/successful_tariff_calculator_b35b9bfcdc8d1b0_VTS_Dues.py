def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate VTS Dues for a vessel based on gross tonnage and port.
    
    VTS charges are based on gross tonnage with different rates for 
    Durban/Saldanha Bay versus other ports, subject to minimum fee.
    """
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    port = vessel_data.get('port', '')
    
    # Use fallback for missing gross tonnage
    if gross_tonnage is None:
        gross_tonnage = 0
    
    # Determine rate based on port
    if port.lower() in ['durban', 'saldanha bay']:
        rate_per_gt = 0.65  # ZAR per GT for Durban and Saldanha Bay
    else:
        rate_per_gt = 0.54  # ZAR per GT for all other ports
    
    # Calculate VTS charge
    calculated_charge = gross_tonnage * rate_per_gt
    
    # Apply minimum fee
    minimum_fee = 235.52  # ZAR
    vts_dues = max(calculated_charge, minimum_fee)
    
    return {
        'VTS Dues': vts_dues
    }
