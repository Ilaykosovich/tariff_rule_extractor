def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate VTS Dues based on vessel gross tonnage and port location.
    
    Args:
        vessel_data: Dictionary containing vessel information
        
    Returns:
        Dictionary with calculated VTS Dues
    """
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    port = vessel_data.get('port')
    
    # Handle missing gross tonnage
    if gross_tonnage is None:
        raise ValueError("Gross tonnage is required for VTS Dues calculation")
    
    # Handle missing port
    if port is None:
        raise ValueError("Port is required for VTS Dues calculation")
    
    # Determine rate based on port
    if port.lower() in ['durban', 'saldanha bay']:
        rate_per_gt = 0.65
    else:
        rate_per_gt = 0.54
    
    # Calculate VTS charge
    vts_charge = gross_tonnage * rate_per_gt
    
    # Apply minimum fee
    minimum_fee = 235.52
    vts_charge = max(vts_charge, minimum_fee)
    
    return {
        'VTS Dues': vts_charge
    }
