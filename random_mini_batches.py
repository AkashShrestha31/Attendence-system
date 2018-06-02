import numpy as np
def random_mini_batches(X,Y,mini_batches,seed):
	batches=[]
	np.random.seed(seed)
	num_mini_batches=int(X.shape[0]/mini_batches)
	permutation=list(np.random.permutation(X.shape[0]))
	X_shuffled=X[permutation,:]
	Y_shuffled=Y[permutation,:]
	for k in range(num_mini_batches):
		mini_batches_X=X_shuffled[k*mini_batches:(k+1)*mini_batches,:]
		mini_batches_Y=Y_shuffled[k*mini_batches:(k+1)*mini_batches,:]
		batch=(mini_batches_X,mini_batches_Y)
		batches.append(batch)
	if X.shape[0]%mini_batches!=0:
		mini_batches_X=X_shuffled[num_mini_batches*mini_batches:,:]
		mini_batches_Y=Y_shuffled[num_mini_batches*mini_batches:,:]
		batch=(mini_batches_X,mini_batches_Y)
		batches.append(batch)
	return batches
