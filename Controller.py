from inputs import *
for device in devices:
	print(device)
while True:
	#events = get_gamepad()
	keys = get_key()
	for key in keys:
	#	if(key.code == 'KEY_K'):
	#		print("SPAMEGG")
		print(key.ev_type, " ", key.code, " ", key.state, "\n...	")
	#for event in events:
	#	print(event.ev_type, " ", event.code, " ", event.state,"\n...")