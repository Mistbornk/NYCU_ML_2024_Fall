k, P0, P1 = 0.5, 0.6, 0.1
eps, diff, cnt = 1e-4, 100, 0
while (diff > eps):
	trial1_w = (k*(P0**3)) / (k*(P0**3)+(1-k)*(P1**3))
	trial2_w = (k*(P0**2)*(1-P0)) / ((k*(P0**2)*(1-P0)) + (1-k)*(P1**2)*(1-P1))
	trial3_w = (k*((1-P0)**3)) / (k*((1-P0)**3) + (1-k)*((1-P1)**3))

	k_new = (trial1_w + trial2_w + trial3_w)/3
	P0_new = (trial1_w*3 + trial2_w*2) / ((trial1_w + trial2_w + trial3_w)*3)
	P1_new = ((1-trial1_w)*3 + (1-trial2_w)*2) / (((1-trial1_w) + (1-trial2_w) + (1-trial3_w))*3)

	diff = (k_new-k) + (P0_new-P0) + (P1_new-P1)
	k = k_new
	P0 = P0_new
	P1 = P1_new
	cnt += 1


print(f"Total iteration to converge: {cnt}")
print(f"k = {k_new}, P0 = {P0_new}, P1 = {P1_new}")






#k = 0.6724247849080648
#P0 = 0.804472967031378
#P1 = 0.04459431929558123
#k = 0.6699821422730127
#P0 = 0.8267597329676039
#P1 = 0.004973362739098089
#k = 0.6701661177601647
#P0 = 0.8289478638394561
#P1 = 6.90766531654463e-05