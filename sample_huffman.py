# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Need some alterations to be able to run with the grace framework: Ex Including the compressor import and class signature

import torch
import numpy as np
# from grace_dl.dist import Compressor
import random
import heapq

# class SampleHuffman(Compressor):
class SampleHuffman():
# #     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()

		self.encoding = {}
		self.reverse_encoding = {}
		self.encoded_document = []
		self.code_length = -1

	def compress(self, quantized_grads):

		unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
		base_frequency = self.base_frequencies(unique_qstates, counts_qstates, quantized_grads)
		sample_frequency = self.sample_frequencies(unique_qstates, quantized_grads)
		sample_no_sparse_huff = self.Huffman_Encode(sample_frequency)

		encodings  = {}
		reverse_encodings = {}
		for i in sample_no_sparse_huff:
			encodings[i[0]] = i[1]
			reverse_encodings[i[1]] = int(i[0])
		
		sample_no_sparse_code_length = 0
		for i in sample_no_sparse_huff:
			sample_no_sparse_code_length += base_frequency[i[0]]*len(i[1])
		
		encoded_doc = []
		# print(encodin)
		# print(sample_no_sparse_huff)
		for i in quantized_grads:
			encoded_doc.append(encodings[str(i)])
		
		# store the encodings in the object member variable
		self.encoding = encodings
		# store the reverse encoding useful for decoding the encoded text
		self.reverse_encoding = reverse_encodings
		# store the encoded document
		self.encoded_document = encoded_doc
		# store the encoded document codelength 
		self.code_length = sample_no_sparse_code_length
		encoded_doc = "".join(encoded_doc)
		return encoded_doc

	def decompress(self):

		decoded_doc = []


		# run codeword
		s = ""

		# traverse the document bit-by-bit and check if current codeword is a valid codeword if yes initialize the run codeword to ""
		for char in self.encoded_document:
			s += char
			try:
				val = self.reverse_encoding[s]
				decoded_doc.append(val)
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

