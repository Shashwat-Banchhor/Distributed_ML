# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Might need some alterations to be able to run with the grace framework

import torch
import numpy as np
from grace_dl.dist import Compressor

import heapq
class RunLengthHuffman(Compressor):
#     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()
		self.code_length = 0
		self.encoded_document = []
		self.reverse_encoding = {}

	def compress(self, quantized_grads):
	

		document, doc_len, frequency =  self.Run_Length_Encode_efficient(quantized_grads)
		

		total_freq = 0
		for key in frequency:
			total_freq += frequency[key]
		
		run_length_huff = self.Huffman_Encode(frequency)
		encodings  = {}
		reverse_encodings = {}
		for i in run_length_huff:
			encodings[i[0]] = i[1]
			reverse_encodings[i[1]] = i[0]
		


		# encode the run length encoded document
		encoded_doc = []
		for i in range(doc_len):
			encoded_doc.append(encodings[document[i]])
		
		encoded_doc  = "".join(encoded_doc)
		
		# calculate the code-length of the run-length huffman code generated
		run_code_length = 0
		for i in run_length_huff:
			run_code_length += frequency[i[0]]*len(i[1])
		
		# store the variable as member variables of the classs
		self.code_length = run_code_length
		self.encoded_document = encoded_doc
		self.reverse_encoding = reverse_encodings
		return encoded_doc



	def decompress(self):
		
		decoded_doc = []
		# run codeword
		s = ""
		# traverse the document bit-by-bit and check if current codeword is a valid codeword if yes initialize the run codeword to ""
		for char in self.encoded_document:
			s += char
			try:
				symbol = self.reverse_encoding[s]
				val = int(symbol.split('c')[0])
				for i in range(int(symbol.split('c')[-1])):
					decoded_doc.append(val)
				s=""
			except:
				continue
		print(decoded_doc)
		return decoded_doc





	def Run_Length_Encode_efficient(self, grads):
		out = []
		index = 0
		s = -10000
		frequency = {}
		for gradient_idx in range(len(grads)):
			if gradient_idx==0:
				s = int(grads[gradient_idx])
				count = 1

			else :
				if (s!= int(grads[gradient_idx])):
					
					out.append(str(s)+'c'+str(count))
					try:
						frequency[str(s)+'c'+str(count)] += 1
					except KeyError as e:
						frequency[str(s)+'c'+str(count)] = 1
					index += 1
					s = int(grads[gradient_idx])
					count = 1

				else:
					count += 1
				
			if (gradient_idx==len(grads)-1):
				out.append(str(s)+'c'+str(count))
				index+=1
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1


		return out, index, frequency

	def Huffman_Encode(self, frequency):
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


