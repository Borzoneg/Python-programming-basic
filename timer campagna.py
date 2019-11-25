ok = False
while not ok:
	min = int(input("Quanti minuti rimangono: "))	
	sec = 0
	print("rimangono solo ", min, " minuti?")

	risp = input()
	if (risp == "y" or risp == "Y"):
		ok = True

tempo = min * 60 + sec
print("premi q per levare un turno, w per levare minuti, e per levare secondi\n")
while tempo > 0:
	print("\n", "{number:02}".format(number = min), ":", "{number:02}".format(number = sec))
	risp = input()
	if risp == 'q':
		tempo = tempo - 6
		min = tempo // 60
		sec = tempo - min * 60
	if risp == 'w':
		x = int(input("quanti minuti vuoi levare?"))
		tempo = tempo - x * 60
		min = tempo // 60
		sec = tempo - min * 60
	if risp == 'e':
		x = int(input("quanti secondi vuoi levare?"))
		tempo = tempo - x
		min = tempo // 60
		sec = tempo - min * 60

print("TEMPO SCADUTO")
		

