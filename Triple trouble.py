def triple_double(num1, num2):

    num1 = str(num1)
    num2 = str(num2)
    print (len(num1))
    print (len(num2))
    i = 0
    k = 0

    for i in range(len(num1)-1):
        if int(num1[i]) == int(num1[i+1]):
            print("k:", k)
            k += 1
            if k == 3:
                ripetuto = num1[i]
                print(ripetuto)
        i += 1

    i = 0
    k = 0

    for i in range (len(num2)-1):
        if int(num2[i]) == int(num2[i+1]):
            print("k:",k)
            k += 1
            if k == 2:
                ripetuto2 = num2[i]
                print(ripetuto2)

    #if ripetuto == ripetuto2:
        #return 1

print (triple_double(451999277, 41177722899))

