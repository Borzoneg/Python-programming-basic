from inputs import *
from playsound import playsound
from multiprocessing import Process
import time

note = {'Q': '0m-072-C5', 'W': '0m-075-D#5', 'E': '0m-078-F#5', 'R': '1f-081-A5'}
processes = []


def play(note):
	note = int(note)
	if note == 1:
		playsound('/home/guglielmo/Documenti/Programmazione/Python/Piano/0m-072-C5.wav')
	if note == 2:
		playsound('/home/guglielmo/Documenti/Programmazione/Python/Piano/0m-075-D#5.wav')


def prendi_tasto():
	keys = get_key()
	for key in keys:
		if(key.code == 'KEY_Q' and key.state == 1):	
			return('q')
		if(key.code == 'KEY_W' and key.state == 1):	
			return('w')
		if(key.code == 'KEY_E' and key.state == 1):	
			return('e')
		if(key.code == 'KEY_R' and key.state == 1):	
			return('r')
		if(key.code == 'KEY_C' and key.state == 1):	
			return('c')
		if(key.code == 'KEY_B' and key.state == 1):	
			return('b')
		if(key.code == 'KEY_Y' and key.state == 1):	
			return('y')
		if(key.code == 'KEY_U' and key.state == 1):	
			return('u')
		if(key.code == 'KEY_I' and key.state == 1):	
			return('i')
		if(key.code == 'KEY_O' and key.state == 1):	
			return('o')

# a = time.time()

# p1 = Process(target=play, args=('1'))
# p2 = Process(target=play, args=('0'))

# p1.start()
# p2.start()

# p1.join()
# p2.join()

# print(time.time() - a)
i = 0
while True:
	tasto = prendi_tasto()
	process1 = Process(target=play,args='0')
	process2 = Process(target=play,args='0')
	
	if (tasto == 'q'):
		process1 = Process(target=play,args='1')
		if not i:
			i = 1
		else:
			process2 = Process(target=play,args='2')
	if (tasto == 'w'):
		print("")
	if (tasto == 'e'):
		print("")
	if (tasto == 'r'):
		print("")
	if (tasto == 'c'):
		print("")
	if (tasto == 'b'):
		print("")
	if (tasto == 'y'):
		print("")
	if (tasto == 'u'):
		print("")
	if (tasto == 'i'):
		print("")
	if (tasto == 'o'):
		print("")

	process1.start()
	process2.start()

	process1.join()
	process2.join()