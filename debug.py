from JunQi import JunqiEnv, PieceType
env = JunqiEnv()
csmap = env.reset()

for i in range(5):
	for j in range(6):
		print(csmap[i][j],end=' ')
	print()
