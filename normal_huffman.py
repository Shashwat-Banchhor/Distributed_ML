# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Need some alterations to be able to run with the grace framework: Ex Including the compressor import and class signature
import torch
import numpy as np
# from grace_dl.dist import Compressor

import heapq

# class NormalHuffman(Compressor):
class NormalHuffman():
# #     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()

		self.encoding = {}
		self.reverse_encoding = {}
		self.encoded_document = []
		self.code_length = -1

	def compress(self, quantized_grads):

		# get the freuqencies of the different quantization states in the quantized gradients
		unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
		base_frequency = self.base_frequencies(unique_qstates, counts_qstates, quantized_grads)

		# create huffma frequency using the freuqencies
		base_huff = self.Huffman_Encode(base_frequency)
		
		base_code_length = 0
		max_len = -1
		

		# create the encodings and the reverse encodings
		encodings = {}
		reverse_encoding = {}
		for i in base_huff:
			base_code_length += base_frequency[i[0]]*len(i[1])
			encodings[i[0]] = i[1]
			reverse_encoding[i[1]] = int(i[0])
		

		# encode the document
		encoded_doc = []
		for i in quantized_grads:
			encoded_doc.append(encodings[str(i)])
		

		# store the encodings in the object member variable
		self.encoding = encodings
		# store the reverse encoding useful for decoding the encoded text
		self.reverse_encoding = reverse_encoding
		# store the encoded document
		self.encoded_document = encoded_doc
		# store the encoded document codelength 
		self.code_length = base_code_length
		


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





