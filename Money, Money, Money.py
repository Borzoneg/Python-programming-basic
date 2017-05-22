def calculate_years(principal, interest, tax, desired):
    y = 0
    while principal < desired :
        interestC = principal * interest 
        principal = principal + (interestC - interestC * tax)
        y =  y + 1
    return y

print(calculate_years(1000, 0.05, 0.18, 1100))
        
