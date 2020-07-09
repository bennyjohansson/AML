import numpy as np
import loadmatrix

#--
# THIS FILE LOADS AND PREPARES DATA FROM EKOSM TRANSACTION FILES
#--

#source ~/tensorflow/bin/activate

# Call functions to load data from Ekosim to temporary matrix
A = loadmatrix.load_temp_data()

# Call functions to prepare, clean, shuffle and re-lable data 
number_of_transactions = 200000
X, Y, mu, sigma = loadmatrix.prepare_data_full(A, number_of_transactions, normalize_my_data = True)

##Defining training and test set
X_train = X[:,0:150000]
Y_train = Y[:,0:150000]
X_test = X[:,150000:200000]
Y_test = Y[:,150000:200000]

print X_train[:, 1:10]

