def friend(x):
    friends = []
    for i in range(len(x)):
        if len(x[i]) == 4:
            friends.append(x[i])
    return friends

print (friend(["Ryan", "Kieran", "Mark"]))
            
