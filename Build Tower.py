def tower_builder(n_floors):
    i = 0
    j = 1
    tower = []
    while i < n_floors:
        floor = (" " * (n_floors - (i+1))) + "*" * j + (" " * (n_floors - (i+1)))
        tower.append(floor)
        j = j+ 2
        i = i+ 1
        
    return tower

print(tower_builder(3))
