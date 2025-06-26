from JunQi import JunqiEnv, PieceType
env = JunqiEnv()
csmap = env.reset()

print("#:")
for i in range(5):
	for j in range(6):
		print(csmap[i][j],end=' ')
	print()

print("$:")
print(env.output())
print()