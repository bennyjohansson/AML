import numpy as np
import tensorflow as tf

#source ~/tensorflow/bin/activate

#--
# THIS FILE CONTAINS THE FUNCTIONS TO LOAD AND CLEAN DATA
#--

#LOAD THE RAW TRANSACTION FILE
def load_temp_data():

	A = load_ekosim_file("transactions_full.txt")
	#np.loadtxt("transactions_full.txt", dtype={'names': ('year', 'amount', 'cp1', 'cp2', 'type'), 'formats': ('f4', 'f4', 'S23', 'S23', 'S23')})	

	return A

#LOAD THE RAW TRANSACTION FILE
def load_ekosim_file(filename):

	A = np.loadtxt(filename, dtype={'names': ('year', 'amount', 'cp1', 'cp2', 'type', 'fraud'), 'formats': ('f4', 'f4', 'S23', 'S23', 'S23', 'i4')})

	return A

#MAIN FUNCTION THAT CALLS ALL THE HELP FUNCTIONS TO PREPARE TRANSACTION DATA
def prepare_data_full(myMatrix, number_of_lines, normalize_my_data = True):

	myMatrix_trans = transform_data(myMatrix, number_of_lines)
		
	X, Y = clean_and_shuffle_data(myMatrix_trans)
	
	
	train_y = Y
	
	if (normalize_my_data):
		train_x, mu, sigma = normalize_data(X)
	else:
		train_x = X
		mu = 0
		sigma = 0
	
	return train_x, train_y, mu, sigma

#SELECTING PART OF THE DATASET AND CLEANS
def transform_data(myMatrix, number_of_lines):

	A = myMatrix

	cp1 = 0
	
	if number_of_lines == 0:
		number_of_lines = A.shape[0]
		
	C = np.zeros((number_of_lines,6))
	

	for i in range(0, number_of_lines):

		C[i,0] = A[i][0]
		C[i,1] = A[i][1]
		C[i,5] = A[i][5]
		
		cp_labels = load_counterpatry_labels()
		
		cp1str = A[i][2]
		cp2str = A[i][3]
		cp1int = 10
		cp2int = 10
		
		if(cp1str.find('Consumer') >= 0):
			cp1int = 9
		else:
			cp1int = cp_labels[cp1str]
		
		if(cp2str.find('Consumer') >= 0):
			cp2int = 9
		else:
			cp2int = cp_labels[cp2str]
					
		C[i,2] = cp1int
		C[i,3] = cp2int
		

		t_labels = load_transaction_labels()
		
		C[i,4] = t_labels[A[i][4]]
		
	return C

#RANDOM SHUFFEL OF DATA AND SOME CLEANUP
def clean_and_shuffle_data(myMatrix):

	Y = np.zeros((myMatrix.shape[0], 1))
	X = np.zeros((myMatrix.shape[0], 4))
	
	myMatrix_shuff = tf.random_shuffle(myMatrix)
		
	with tf.Session() as sess:
		myMatrix_shuff = sess.run(myMatrix_shuff)
		
	Y[:,0] = myMatrix_shuff[:,5]
	
	Y = Y.T
	
	X = myMatrix_shuff[:,1:5]
	X = X.T
	
	return X,Y

#NORMALIZING THE MATRIX
def normalize_data(myMatrix):

	mu = np.zeros((myMatrix.shape[0],1))
	sigma = np.zeros((myMatrix.shape[0],1))
	
	mu[:,0] = np.mean(myMatrix, axis=1)
	
	sigma[:,0] = np.var(myMatrix, axis=1)
	
	myMatrix = myMatrix - mu
	
	myMatrix /= sigma
	
	return myMatrix, mu, sigma


#CONVERT TRANSACTION TYPE STRINGS TO NUMBER
def load_transaction_labels():

	mylabels = {'Inventory':0, 'Purchase' : 1, 'Dividend' : 2, 'Dividends' : 2, 'Salary' : 3, 'Interest' : 4, 'Amortization' : 5, 'Loan' : 6, 'Investment' : 7, 'Deposit' : 8}
	
	return mylabels
	
#CONVERT COUNTERPARTY TYPE STRINGS TO NUMBER
def load_counterpatry_labels():

	mylabels = {'Company':0, 'Bank' : 1, 'market' : 2, 'Market' : 2, 'johansson_och_johansson' : 3, 'benny_enterprises' : 4, 'limpan_AB' : 5, 'bempa_co' : 6, 'bempa_AB' : 7,'benny_inc' : 8, 'Consumer': 9}
	
	return mylabels

