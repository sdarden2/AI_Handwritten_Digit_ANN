#Utility functions for any class or module
import numpy as np

def sig(x):
	
	res = 1.0/(1+ np.e**(-x))
	
	return res
	
def parse_inputs(f,testing_set=False):
	fr = open(f,'r')
	text = fr.read()
	fr.close()
	
	if testing_set:
		iters = 1797
	else:
		iters = 3823
	text = text.replace('\n',' ')
	text = text.replace(',',' ')
	l = text.split(' ')
	for k in range(l.count('')):
		l.remove('')
	l = [int(i) for i in l]
	print len(l)
	
	matrices = []
	num_list = []
	for i in range(iters):
		mini_list = l[i*65:(i+1)*65]
		num_list.append(mini_list.pop())
		mat = []
		for ind in range(1,9):
			mat.append(mini_list[8*(ind-1):8*ind])
		matrices.append(mat)
	return matrices,num_list
	
def print_matrix(mat):
	for i in mat:
		print i
	
