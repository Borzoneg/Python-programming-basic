def likes(names):
        if len(names) == 0:
            return "No one like this"
        elif len(names) == 1:
            return (names[0] + "like this")
        elif len(names) == 2:
            return (names[0] + " and " + names[1] + " like this")
        elif len(names) == 3:
            return (names[0] + ", " + names[1] + " and " + names[2] + " like this")
        else:
            return(names[0] + ", " + names[1] + " and other " + str((len(names) - 2)) + " like this")


print(likes(["Jacob", "Alex", "asd", "Ayeye", "asdsa", "poi"]))