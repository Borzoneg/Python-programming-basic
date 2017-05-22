def duplicate_count(text):
    a = []
    b = []
    i = 0
    for char in text:
        if char not in b:
            if char in a:
                i += 1
                b.append(char)
            else:
                a.append(char)
    print(i)
    return i

duplicate_count('aabbcde')
duplicate_count('aabbcdeB')
duplicate_count('Indivisibilities')
duplicate_count('aa11')

