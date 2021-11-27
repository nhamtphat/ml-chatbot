import reading

def chat():
	print("Start Talking with the bot(type quit to stop!")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		print(reading.response(inp, 1))

chat()
