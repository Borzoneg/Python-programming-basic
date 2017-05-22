def longest_consec(strarr, k):
    i = 0
    j = 0
    while j < len (strarr):
        while i < len (strarr) -1:
            if strarr[i+1] > strarr [i]:
                a = strarr[i+1]
                strarr[i+1] = strarr [i]
                strarr [i] = a
            i = i+1
        j = j+1
    print (strarr)
    
while True:
    x = input("Metti un numero: ")
    longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2)
