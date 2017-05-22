def crypt():
    while True:
        try:
            key = int(input("Inserisci la chiave: "))
            assert key != 0 and key % 26 != 0
            break
        except ValueError:
            print("Non hai inserito una chiave valida, ritenta")
        except AssertionError:
            print("Una chiave con valore 0 o multiplo di 26 non cripterà il tuo messaggio, inserisci una chiave valida")

    msg = input("Inserisci il messaggio da criptare: ")
    msg = msg.lower()

    lton = {" ": 0.5, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11,
            "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22,
            "w": 23, "x": 24, "y": 25, "z": 26}
    ntol = {0.5: " ", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i", 10: "j", 11: "k",
            12: "l", 13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r", 19: "s", 20: "t", 21: "u", 22: "v",
            23: "w", 24: "x", 25: "y", 26: "z"}

    msgn = []
    msgnk = []
    msgc = []

    for char in msg:
        msgn.append(lton[char])

    for num in msgn:
        msgnk.append(num + key)

    for num in msgnk:
        if num > 26:
            num -= 26
        if num == (key + 0.5):
            num = 0.5
        if num <= 0:
            num = 27 + key
        msgc.append(ntol[num])
    msgc = "".join(msgc)  # convertire una lista in string
    print(msgc)


while True:
    crypt()
