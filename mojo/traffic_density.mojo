fn calculate_total(
    cars: Int,
    motorcycles: Int,
    buses: Int,
    trucks: Int,
    bicycles: Int
) -> Int:
    return cars + motorcycles + buses + trucks + bicycles


fn calculate_density(total: Int) -> String:
    if total <= 3:
        return "LOW"
    elif total <= 7:
        return "MEDIUM"
    else:
        return "HIGH"


fn main():
    # Vehicle counts from traffic monitoring system
    var cars: Int = 5
    var motorcycles: Int = 1
    var buses: Int = 2
    var trucks: Int = 1
    var bicycles: Int = 0

    # Calculate total vehicles
    var total = calculate_total(
        cars,
        motorcycles,
        buses,
        trucks,
        bicycles
    )

    # Calculate traffic density
    var density = calculate_density(total)

    # Display results
    print("AI Traffic Monitoring System")
    print("----------------------------")

    print("Cars:", cars)
    print("Motorcycles:", motorcycles)
    print("Buses:", buses)
    print("Trucks:", trucks)
    print("Bicycles:", bicycles)

    print("----------------------------")
    print("Total Vehicles:", total)
    print("Traffic Density:", density)