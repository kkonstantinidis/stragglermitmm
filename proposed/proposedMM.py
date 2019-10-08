#!/usr/bin/env python
'''

'''

from mpi4py import MPI
import numpy as np
import random
import threading
import time

##################### Parameters ########################

# Change to True for more accurate timing, sacrificing performance
barrier = True

#Set to 0 for no straggling, 1 for straggling via the loop() function, 2 for straggling via sleeping and 3 for struggling via extra computation
straggling = 3
straggling_time = 30

#Straggling function
def loop():
  t = time.time()
  while time.time() < t + straggling_time:
    a = 1 + 1

# Use one master and N workers
N = 10

#Number of stragglers
N_S = 0

# Matrix division
m = 2
n = 2
p = 2

#Set to 1 to load pregenerated A,B from current directory or 0 to generate them. Note that parameters of this script and of generateAB.py should match.
#Also, matrices have to be in a folder "pregenerateAB" in parent directory
loadAB = 1

#Bound of the elements of the input matrices i.e. those should be in [0,...,B]
B = 50

#Input matrix size - A: s by r, B: s by t.
s = 8000
r = 8000
t = 8000

#Recovery threshold
T = m*n

#This is the parameter s of the IEEE communcation letters paper i.e. we need to make sure that s_cod >= 2*L = 2*(B^2*s + 1)
s_cod = 67108864 #For 8000x8000, B=50

#Tolerance used for rounding
tol = 1e-14 

# Values of x_i used by workers. They are the Chebysev points of the 1st kind (real).
var = np.linspace(-1, 1, N, True).astype(np.float64)

#########################################################

comm = MPI.COMM_WORLD

#Check for wrong number of MPI processes
if comm.size != N+1:
	print("The number of MPI processes mismatches the number of workers.")
	comm.Abort(1)
	
if comm.rank == 0:
	# Master
	print "Running with %d processes:" % comm.Get_size()

	print "ISU, N=%d workers, N_S=%d straggler(s), s=r=t=%d, B=%d, s_cod=%d, real" % (N, N_S, s, B, s_cod)
	
	#Decide and broadcast chosen stragglers
	if straggling != 0:
		stragglers = set()
		ctr = 0
		while ctr < N_S:
			rand = random.randint(1, N)
			if rand not in stragglers:
				stragglers.add(rand) 
				ctr += 1

		for i in range(N_S):
			straggler = stragglers.pop()
			for j in range(N):
				comm.send(straggler, dest=j+1, tag=7)

	bp_start = time.time()
	
	#Create random matrices or load them from files. Now it doesn't make sense to use np.int64.
	if loadAB == 0:
		A=np.matrix(np.random.random_integers(0,B,(s,r)))
		B=np.matrix(np.random.random_integers(0,B,(s,t)))
	elif loadAB == 1:
		A = np.load('../pregenerateAB/A.npy')
		B = np.load('../pregenerateAB/B.npy')
	
	#64-bit precision is inherited here
	Ah=np.split(A,p)
	Bh=np.split(B,p)
	
	Ahv = []
	Bhv = []
	for i in range(p):
		Ahv.append(np.split(Ah[i], m, axis=1))
		Bhv.append(np.split(Bh[i], m, axis=1))
	
	# Encode the matrices
	Aenc = []
	Benc = []
	for i in range(N):
		Aenc.append(Ahv[0][0] + Ahv[1][0]*1.0/s_cod + Ahv[0][1]*var[i] + Ahv[1][1]*1.0/s_cod*var[i])    
		Benc.append(Bhv[0][0] + Bhv[1][0]*s_cod + Bhv[0][1]*pow(var[i], 2) + Bhv[1][1]*s_cod*pow(var[i], 2))  		
	
	bp_end = time.time()
	print "Time spent for pre-processing is: %f" % (bp_end - bp_start)
	
	# Initialize return dictionary
	Rdict = []
	for i in range(N):
		Rdict.append(np.zeros((r/m, t/n), dtype=np.float64))

	# Start requests to send and receive
	reqA = [None] * N
	reqB = [None] * N
	reqC = [None] * N
  
	bp_start = time.time()
  
	for i in range(N):
		reqA[i] = comm.Isend([Aenc[i],MPI.DOUBLE], dest=i+1, tag=15)
		reqB[i] = comm.Isend([Benc[i],MPI.DOUBLE], dest=i+1, tag=29)
		reqC[i] = comm.Irecv([Rdict[i],MPI.DOUBLE], source=i+1, tag=42)

	MPI.Request.Waitall(reqA)
	MPI.Request.Waitall(reqB)
  
	# Optionally wait for all workers to receive their submatrices, for more accurate timing
	if barrier:
		comm.Barrier()

	bp_sent = time.time()
	print "Time spent sending all messages is: %f" % (bp_sent - bp_start)

	Crtn = [None] * N
	lst = []
	#Wait for the mn fastest workers
	for i in range(T):
		j = MPI.Request.Waitany(reqC)
		lst.append(j)
		Crtn[j] = Rdict[j]
	bp_received = time.time()
	print "Time spent waiting for %d workers %s is: %f" % (T, ",".join(map(str, [x+1 for x in lst])), (bp_received - bp_sent))
	
	bp_start = time.time()
	
	#This is to prevent segmentation fault for large matrices
	if barrier:
		comm.Barrier()
	
	#Receive computation time from workers
	comp_time = np.array([[0] for i in range(N)]).astype(np.float64)
	for i in range(N):
		comp_time[i] = comm.recv(source=i+1, tag=50)
	print "The average worker computation time is: %f" %(np.sum(comp_time)/N)
	print "The T-th fastest worker is: %f" % (np.partition(np.asarray(np.reshape(comp_time, (N,))), T-1)[T-1])
	
	bp_end = time.time()
	barrier_time = bp_end - bp_start
	
	print ""
	print("Starting decoding...")
	
	bp_start = time.time()
	
	#Compute the inverse of Vandermonde manually based on NASA paper. Do not change the ones_like()
	L_inv = np.ones_like(np.matrix([[0]*(T) for i in range(T)])).astype(np.float64)
	for i in range(T):
		for j in range(T):
			if i < j:
				L_inv[i,j] = 0
			elif i > 0:
				ran = range(i+1)
				del ran[j]
				prod_terms = np.array([1.0/(var[lst[j]]-var[lst[k]]) for k in ran]).astype(np.float64)
				
				#1st way
	#            L_inv[i,j] = np.prod(prod_terms)
				
				#2nd way
				prod_tmp = 1
				for k in range(i):
					prod_tmp = prod_tmp*prod_terms[k]
				L_inv[i,j] = prod_tmp
				
	U_inv = np.empty_like(np.matrix([[0]*(T) for i in range(T)])).astype(np.float64)
	for i in range(T):
		for j in range(T):
			if i == j:
				U_inv[i,j] = 1
			elif j == 0:
				U_inv[i,j] = 0
			else:
				if i == 0:
					U_inv[i,j] = -U_inv[i,j-1]*var[lst[j-1]]
				else:
					U_inv[i,j] = U_inv[i-1,j-1]-U_inv[i,j-1]*var[lst[j-1]]
	
	#We need it to be an array for the 3D idea to work
	V_inv = np.array(np.dot(U_inv, L_inv))
	
	bp_end = time.time()
	print "Time spent for inverting Vandermonde is: %f" % (bp_end - bp_start)
	
	bp_start = time.time()
	
	#Keep returned values only of the fastest T workers  
	#Crtn_cat is the vectorized results of the workers i.e. we vectorize each result into 1 row of the Crtn_cat 
	#of length r/m*t/n.
	Crtn_vec = np.empty([T,r/m*t/n]).astype(np.float64)
	for i in range(T):
		Crtn_vec[i] = np.reshape(Crtn[lst[i]], r/m*t/n)
	
	bp_end = time.time()
	print "Time spent for vectorization is: %f" % (bp_end - bp_start)
	
	bp_start = time.time()
	
	#Decode only the 4 useful terms (blocks) of C
	#Do element-wise polynomial interpolation by Vandermonde inversion
	#Decoded values should be real
	Crtn_vec = np.round(np.dot(V_inv,Crtn_vec)).astype(np.int64)
	
	bp_end = time.time()
	print "Time spent for multiplying with the Vandermonde inverse is: %f" % (bp_end - bp_start)
	
	bp_start = time.time()
	
	#Devectorize and convert Crtn_vec to a list of matrices	
	C_hat = []
	for i in range(m*n):
		
		#We need all 4 terms
		C_hat.append(np.reshape(Crtn_vec[i], (r/m,t/n)).astype(np.int64))
	
	bp_end = time.time()
	print "Time spent for devectorization is: %f" % (bp_end - bp_start)
	
	bp_start = time.time()
	
	#Decode and concatenate column by column. Save returned product to file.
	C = np.empty((r,0), int)
	for i in range(n):
		cur_col = np.empty((0, t/n), int)
		
		#Construct column
		for j in range(m):				
			
			#Compute the term Cij of expression \hat{Cij} = Cij + "*"s + ... + "*"s^{p-1} i.e. the remainder of the division
			mod_s_cod = np.fmod(C_hat[n*i+j], s_cod)

			#If Cij has absolute value greater than s_cod^{\ell}/2 it will need to be adjusted
			logic = abs(mod_s_cod) > 0.5*s_cod

			#Convert True-False to 1-0
			logic = logic*1

			#Decode Cij and append it to the final result
			Cij = mod_s_cod - np.multiply(logic, np.sign(mod_s_cod)*s_cod)
			cur_col = np.append(cur_col, Cij, axis=0)	
		
		#Concatenate column 
		C = np.append(C, cur_col, axis=1)
	
	bp_end = time.time()
	print "Time spent for modulo and final decoding is: %f" % (bp_end - bp_start)
	
	bp_start = time.time()
	
	#Test
	np.save('C', C)
	
	bp_end = time.time()
	store_time = bp_end - bp_start
	print "Time spent for storing C to drive is: %f" % (store_time)
	
	bp_done = time.time()
	print "Time spent decoding is: %f" % (bp_done - bp_received - barrier_time - store_time)
	
else:
	# Worker
	#Each slave will receive all the straggler ranks but needs to keep only himself (if he is a straggler)
	straggler = 0
	if straggling != 0:
		for i in range(N_S):
			straggler_tmp = comm.recv(source=0, tag=7)
			if straggler_tmp == comm.rank:
				straggler = straggler_tmp

	# Receive split input matrices from the master
	Ai = np.empty_like(np.matrix([[0]*(r/m) for i in range(s/p)])).astype(np.float64)
	Bi = np.empty_like(np.matrix([[0]*(t/n) for i in range(s/p)])).astype(np.float64)
		
	rA = comm.Irecv(Ai, source=0, tag=15)
	rB = comm.Irecv(Bi, source=0, tag=29)

	rA.wait()
	rB.wait()
	
	if barrier:
		comm.Barrier()
	
	wbp_received = time.time()
	
	if straggling != 0:
		if straggler == comm.rank:
			print "I am worker %d and I am struggling." % comm.rank
			if straggling == 1:
				thread = threading.Thread(target=loop)
				thread.start()
			elif straggling == 2:
				time.sleep(straggling_time)
			elif straggling == 3:
				Ci_trash = np.matmul(np.transpose(Ai), Bi)

	#We compute A^T*B
	
	#np.matmul() is faster than np.dot()
	Ci = np.matmul(np.transpose(Ai), Bi)
		
	wbp_done = time.time()
	print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)
	
	#Using numpy
	sC = comm.Isend(Ci, dest=0, tag=42)
	sC.Wait()
	
	#This is to prevent segmentation fault for large matrices
	if barrier:
		comm.Barrier()

	comp_time = wbp_done - wbp_received
	comm.send(comp_time, dest=0, tag=50)