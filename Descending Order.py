def Descending_Order(num):
    list1 = repr(num)
    list1 = list(list1)
    i = 0
    j = 0

    while i < len(list1):
        while j < (len(list1) - (i+1)):
            if int(list1 [j]) < int(list1 [j+1]):
                a = list1 [j]
                list1 [j] = list1 [j+1]
                list1 [j+1] = a
                j = j + 1
            else:
                j = j + 1
        i = i + 1
        j = 0
    return int(''.join(list1))


print(Descending_Order(123566789))
