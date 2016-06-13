#ANN #2 
from ANNutils import *
import numpy as np
import random
from matplotlib import pyplot as plt

#Code loosely based on information from http://neuralnetworksanddeeplearning.com/chap1.html
#Idea to use matrices to represent the ANN
INPUT_SIZE = 8**2
HIDDEN_SIZE = 38 #38 works well
OUT_SIZE = 10
LEARNING_FACTOR = .1 #.1 works well
TRAIN_SIZE = 3823
TEST_SIZE = 1797
#NO biases
#Error gets stuck at .5
#value is stuck at 178
class ANN:
	def __init__(self,input_size,hidden_size,output_size,input_matrix,input_values,test_matrix,test_values):
		self.insize = input_size
		self.hsize = hidden_size
		self.outsize = output_size
		self.input_matrix = input_matrix
		self.input_vals = input_values
		self.test_matrix = test_matrix
		self.test_vals = test_values
		self.build()
		
	def build(self):
		mat= np.array(self.input_matrix)
		
		
		self.hidden_vect = np.zeros(self.hsize)
		self.out_vect = np.zeros(self.outsize)
		
		self.weights1 = np.random.randn(self.insize,self.hsize)
		self.weights2 = np.random.randn(self.hsize,self.outsize)
		
	def feed(self,index,train=True):
		#index is index of input_matrix to choose for inputs
		if train:
			in_mat = np.reshape(self.input_matrix[index],(1,INPUT_SIZE))
		else:
			in_mat = np.reshape(self.test_matrix[index],(1,INPUT_SIZE))
		
		
		hid_res = in_mat.dot(self.weights1)
		
		hid_res = sig(hid_res)
		
		out_res = hid_res.dot(self.weights2)
		out_res = sig(out_res)
		
		
		self.in_vect = in_mat
		self.hidden_vect = hid_res
		self.out_vect = out_res
		
	def train(self,iters):
		for iter in range(iters):
			#total_delta_w1 = np.zeros((self.insize,self.hsize))
			#total_delta_w2 = np.zeros((self.hsize,self.outsize))
			for mat_index in range(len(self.input_matrix)):
				self.feed(mat_index)
				
				expected_vect = np.zeros(self.outsize)
				expected_vect[self.input_vals[mat_index]] = 1.0
				
				#Backpropogate
				#Get output error
				out_err_vect = (expected_vect - self.out_vect)*self.out_vect*(1.0-self.out_vect)
				
				#Get hidden node error
				sm = self.weights2.dot(out_err_vect.transpose())
				hidden_err = sm*(self.hidden_vect.transpose())*(1.0-self.hidden_vect.transpose())
				
				#Need L2^T*out_err_vect
				w2_delta = self.hidden_vect.transpose().dot(out_err_vect)
				w1_delta = (hidden_err.dot(self.in_vect)).transpose()
				
				#total_delta_w1 += w1_delta
				#total_delta_w2 += w2_delta
				ERROR = self.calc_error(self.input_vals[mat_index],self.out_vect)
				
			#Set new weights
			#This is giving me the highest number so far
				self.weights1 = self.weights1 + (w1_delta*LEARNING_FACTOR)
				self.weights2 = self.weights2 + (w2_delta*LEARNING_FACTOR)
			
	def test(self):
		num_right = 0
		for i in range(len(self.test_matrix)):
			val = self.test_vals[i]
			mat = self.test_matrix[i]
			
			self.feed(i,False)
			out_list = self.out_vect.tolist()[0]
			maxval = max(out_list)
			maxindex = out_list.index(maxval)
			if i%578 == 0:
				print "WERE FEEDING IN MATRIX:\n"
				print str(np.reshape(mat,(1,64))) + "\n"
				print "\n\nFOR OUTLIST: \n"
				for k in range(len(out_list)):
					print str(k) + " : " + str(out_list[k])
				print "EXPECTED: " + str(self.test_vals[i])
				print "GOT: " + str(maxindex)
				
			if maxindex == self.test_vals[i]:
				num_right+=1
		print "NUM RIGHT: %d" %num_right
		print "OUT of %d" %(len(self.test_matrix))
		print "PERCENTAGE: %.08f" %(num_right*100.0/(len(self.test_matrix)))
		return num_right
		
	def calc_error(self,exp_val,output):
		#Calcs the current error in after the feedforward
		#exp_val is the expected output values
		#output is the actual output vector
		exp_out = np.zeros(10)
		exp_out[exp_val] = 1.0
		
		res = sum((exp_out-output[0])**2)/2.0
		return res
		
def main():
	matrices,num_list = parse_inputs("optdigits_train.txt")
	test_mat,test_nums = parse_inputs("optdigits_test.txt",True)
	n = ANN(INPUT_SIZE,HIDDEN_SIZE,OUT_SIZE,matrices,num_list,test_mat,test_nums)
	#6 was giving decent results
	
	train_list = [1,2,3,5,6,10,20,50,100,200,400,1000,1500]
	#n.train(800)
	#n.test()
	res_list = []
	for k in train_list:
		n.train(k)
		res = n.test()
		res_list.append(res)
	print train_list
	print res_list
	
	res_list = np.array(res_list)
	print res_list/float(TEST_SIZE)*100.0
	x = np.array(train_list)
	plt.ylim(1000,TEST_SIZE + 5)
	plt.plot(x,res_list)
	plt.show()
if __name__ == "__main__":
	main()
	