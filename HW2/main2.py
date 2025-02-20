from utils import C, B

if __name__ == '__main__':
	with open('testfile.txt', 'r') as file:
		testcase = file.readlines()
	# alpha, beta
	a = int(input('input a: ')) 		
	b = int(input('input b: '))

	for i in range(len(testcase)): 
		line = testcase[i].strip()
		# total N trials
		N = len(line) 					
		m = 0		  
		
		# m successes
		for char in line:
			m += 1 if char=='1' else 0  
		# P( x | theta)
		P = m / N  						
		likelihood = C(N, m) * (P**m) * ((1-P)**(N-m)) 
		prior = (P**(a-1)) * ((1-P)**(b-1)) * (1/B(a, b))

		print(f'case {i + 1}: {line}')
		print(f'Likelihood: {likelihood}')
		print(f'Beta prior:\ta = {a}\tb = {b}')
		
		# beta posterior
		a += m
		b += (N - m)
		
		print(f'Beta posterior:\ta = {a}\tb = {b}\n')


		