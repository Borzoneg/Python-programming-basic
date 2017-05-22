def goodVsEvil(good, evil):
    good = good.split()
    evil = evil.split()
    goodL = []
    evilL = []
    pGood = 0
    pEvil = 0

    for i in range(len(good)):
        goodL.append(int(good[i]))
        i += 1

    for i in range(len(evil)):
        evilL.append(int(evil[i]))
        i += 1

    pointG = [1, 2, 3, 3, 4, 10]
    pointE = [1, 2, 2, 2, 3, 5, 10]

    for i in range (len(goodL)):
        pGood += goodL[i] * pointG[i]
    for i in range (len(evilL)):
        pEvil += evilL[i] * pointE[i]
    if pGood < pEvil:
        return "Battle Result: Evil eradicates all trace of Good, Evil should win"
    if pGood > pEvil:
        return "Battle Result: Good triumphs over Evil, Good should win"
    else:
        return "Battle Result: No victor on this battle field, should be a tie"

print(goodVsEvil('1 0 0 0 0 0', '1 0 0 0 0 0 0'))

