import math

def calculate(vessel_data: dict) -> dict[str, float]:
    """
    Calculate Port Dues for vessels in Durban port.
    
    Port dues are charged from entrance inwards to entrance outwards,
    with basic fee plus daily fees based on gross tonnage.
    """
    
    # Extract required parameters
    gross_tonnage = vessel_data.get('technical_specs', {}).get('gross_tonnage')
    days_alongside = vessel_data.get('operational_data', {}).get('days_alongside')
    
    # Use fallbacks if parameters missing
    if gross_tonnage is None:
        gross_tonnage = 0  # Will trigger minimum fee if applicable
    
    if days_alongside is None:
        days_alongside = 1.0  # Default to 1 day minimum stay
    
    # Port dues rates
    basic_fee_rate = 192.73  # ZAR per 100 tons or part thereof
    daily_fee_rate = 57.79   # ZAR per 100 tons per 24 hour period
    
    # Calculate tonnage units (per 100 tons or part thereof - round up)
    tonnage_units = math.ceil(gross_tonnage / 100.0) if gross_tonnage > 0 else 1
    
    # Calculate basic fee
    basic_fee = tonnage_units * basic_fee_rate
    
    # Calculate daily fees (pro rata for partial periods)
    daily_fees = tonnage_units * daily_fee_rate * days_alongside
    
    # Total port dues
    port_dues = basic_fee + daily_fees
    
    # Check for minimum fee (small vessels and pleasure vessels)
    minimum_fee = 470.98
    if port_dues < minimum_fee:
        port_dues = minimum_fee
    
    # Check for less than 12 hours reduction (15% reduction)
    hours_alongside = days_alongside * 24
    if hours_alongside < 12:
        port_dues = port_dues * 0.85  # 15% reduction
    
    # Check for vessels over 30 days (20% surcharge on incremental fee)
    if days_alongside > 30:
        # 20% surcharge applies only to the incremental (daily) portion
        surcharge = daily_fees * 0.20
        port_dues = port_dues + surcharge
    
    return {
        'Port Dues': round(port_dues, 2)
    }
