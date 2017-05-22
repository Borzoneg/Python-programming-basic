def breakChocolate(n, m):
    if n * m <= 0 :
        return 0
    else:
        return n*m - 1

while True:
    x = int(input("Inserisci la x "))
    y = int(input("Inserisci la y "))
    print(breakChocolate(x, y))

