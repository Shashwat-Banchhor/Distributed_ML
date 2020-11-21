# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Need some alterations to be able to run with the grace framework: Ex Including the compressor import and class signature

import torch
import numpy as np
# from grace_dl.dist import Compressor
import random
import heapq

# class SampleHuffmanSparsity(Compressor):
class SampleHuffmanSparsity():
# #     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()

		self.encoding = {}
		self.reverse_encoding = {}
		self.encoded_document = []
		self.code_length = -1


	# compress the qunatized gradients using sample huffman with sparsity 
	# qmin is the minimum possible quantization state
	# qmax is the maximum possible quantization state
	
	def compress(self, qmin, qmax, quantized_grads):


		# the sparsity factor 
		k = 200

		# the sampling factor
		S = 1000

		# Preemptive Smoothing #######################
		total_freq = 0
		sample_frequency = {}
		for key in range(qmin, qmax,1) :
			sample_frequency[str(key)] = 1
			total_freq += 1
		#############################################
		
		# sampling
		for x in range(S):
			### Sampling done uniformly at random ###
			idx=random.randint(0,len(quantized_grads)-1)
			sample_frequency[str(int(quantized_grads[idx]))]+=1
			total_freq += 1


		# get the sample frequencies of all the quantization states
		for key in sample_frequency.keys():
			sample_frequency[key] = sample_frequency[key]/total_freq


		# Add the sparsity for character/symbol 0
		# ### Probability for n-zeros
		for repetitions in range(1,k):   
			sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*(1- sample_frequency[str(0)])
		sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)#*(1- sample_frequency[str(0)])
		del sample_frequency["0"]
		

		# get the huffman codes for the sampled frequencies
		sample_huff = self.Huffman_Encode(sample_frequency)

		# get the encodings and also the reverse encodings(allows to decode)
		encodings  = {}
		reverse_encodings = {}
		for i in sample_huff:
			# print(i)
			encodings[i[0]] = i[1]
			reverse_encodings[i[1]] = i[0]


		
		sample_sparse_code_length = 0
		sample_encoding = {}
		reverse_encodings = {}
		max_len = -1
		for i in sample_huff:
			# print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(30) + i[1])
			sample_encoding[i[0]] = (sample_frequency[i[0]], len(i[1]), i[1])
			reverse_encodings[i[1]] = i[0]
			
		

		# create the encoded document and obtain the codelength
		encoded_doc = []
		document, doc_len, unique_elements, counts_elements =  self.Run_Length_Encode(quantized_grads)
		for run in document:
			sparse = run.split("c")
			sparse[0] = int(sparse[0])
			sparse[1] = int(sparse[1])
			if (sparse[0]==0 and sparse[1]<=k):
				# character of the for 0ck, where c is the separator and k is the number of contigous 0 in the original grad
				sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1])][1]
				encoded_doc.append(sample_encoding[str(sparse[0])+"c"+str(sparse[1])][2])
			else:
				####  We can improve more on this  ######
				if (sparse[0]==0):
					# character of the for 0cr, where c is the separator and r (>k) is the number of contigous 0 in the original grad
					# Ex. 0c(3k+2) = 0ck , 0ck, 0ck, 0c2
					sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(k)][1] * (sparse[1]/k)
					encoded_doc.append(sample_encoding[str(sparse[0])+"c"+str(1)][2] * sparse[1])

					if (sparse[1]%k!=0):
						sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1]%k)][1] 
						encoded_doc.append(sample_encoding[str(sparse[0])+"c"+str(sparse[1]%k)][2])
				else:
					# a qunatization state other than 0

					sample_sparse_code_length += sample_encoding[str(sparse[0])][1] * sparse[1]
					encoded_doc.append(sample_encoding[str(sparse[0])][2] * sparse[1])
		
		encoded_doc = "".join(encoded_doc)
		# store the encodings in the object member variable
		self.encoding = encodings
		# store the reverse encoding useful for decoding the encoded text
		self.reverse_encoding = reverse_encodings
		# store the encoded document
		self.encoded_document = encoded_doc
		# store the encoded document codelength 
		self.code_length = sample_sparse_code_length
		
		return encoded_doc

	def decompress(self):

		print("Entered Decompressing:")
		print(self.encoded_document)
		decoded_doc = []


		# run codeword
		s = ""

		# traverse the document bit-by-bit and check if current codeword is a valid codeword if yes initialize the run codeword to ""
		for char in self.encoded_document:
			s += char
			try:
				val = self.reverse_encoding[s]
				try:
					decoded_doc.append(int(val))
				except:
					for i in range(int(val.split('c')[-1])):
						decoded_doc.append(0)
				s=""
			except:
				continue
		
		return decoded_doc

	
	def base_frequencies(self,unique_qstates, counts_qstates, quantized_grads):
		base_frequency = {}
		cnt = 0
		for x in range(len(unique_qstates)):
			base_frequency[str(int(unique_qstates[x]))] = counts_qstates[x]
			cnt+=1
		while(cnt < len(quantized_grads)):
			cnt+=1
		return base_frequency




	def Huffman_Encode(self,frequency):
		heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
		heapq.heapify(heap)
		while len(heap) > 1:
			low = heapq.heappop(heap)
			high = heapq.heappop(heap)
			for value in low[1:]:
				value[1] = '0' + value[1]
			for value in high[1:]:
				value[1] = '1' +value[1]
			heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
		return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))		



	def sample_frequencies(self, unique_qstates, quantized_grads):   
		
		S = 1000
		# Preemptive Smoothing #######################
		sample_frequency = {}
		for key in unique_qstates :
			sample_frequency[str(int(key))] = 1
		
		for x in range(S):
			### Sampling done uniformly at random ###
			idx = random.randint(0,len(quantized_grads)-1)
			sample_frequency[str(int(quantized_grads[idx]))] +=  1
		return sample_frequency


		

	# run-length encode the quantized gradients
	def Run_Length_Encode(self, grads):
		# maintain the run_length_encoded doc
		out = []
		index = 0
		s = -10000

		for gradient_idx in range(len(grads)):
			if gradient_idx==0:
				s = int(grads[gradient_idx])
				count = 1

			else :
				if (s!= int(grads[gradient_idx])):
					# out.append(str(s))
					# index  += 1
					out.append(str(s)+'c'+str(count))
					index += 1
					s = int(grads[gradient_idx])
					count = 1

				else:
					count += 1
				
			if (gradient_idx==len(grads)-1):
				out.append(str(s)+'c'+str(count))

		out  = np.array(out)
		unique_elements, counts_elements = np.unique(out, return_counts=True)
		
		out = out.tolist()
		return out, len(out), unique_elements, counts_elements



