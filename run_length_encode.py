# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Might need some alterations to be able to run with the grace framework

import torch

from grace_dl.dist import Compressor



class RunLengthEncode(Compressor):
#     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()
		

	def compress(self, grads, qstates):
		"""Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""
		encodings = {}

		# we are using the same encoding to as it is not relevant from what the code length is.
		for i in range(qstates):
			# this should be the binary representation of i represeted in 16-bits
			encodings[i]  = get_16_bit_binary(i)
			assert len(encodings[i]) % 16 == 0 , "encoding len not correct"

		encoded_doc = []
		index = 0
		# invalid intialization value 
		s = -1
		frequency = {}
		for gradient_idx in range(len(grads)):
			if gradient_idx == 0:
				s = int(grads[gradient_idx])
				count = 1

			else :
				if (s!= int(grads[gradient_idx])):
					
					try: 
						# the count of the character in the run
						encoded_doc.append(encodings[s])
					except KeyError as e:
						encodings[s] = get_16_bit_binary(s)
						# the count of the character in the run
						encoded_doc.append(encodings[s])
					try:
						# the character in the run
						encoded_doc.append(encodings[count])
					except KeyError as e:
						
						encodings[count] = get_16_bit_binary(count)
						# the character in the run
						encoded_doc.append(encodings[count])
					
					# assign s to a new character
					index += 1
					s = int(grads[gradient_idx])
					count = 1

				else:
					count += 1
				
			if (gradient_idx==len(grads)-1):
				try: 
					# the count of the character in the run
					encoded_doc.append(encodings[s])
					# the character in the run
					encoded_doc.append(encodings[count])
				except KeyError as e:
					encodings[s] = get_16_bit_binary(s)
					encodings[count] = get_16_bit_binary(count)

					# the count of the character in the run
					encoded_doc.append(encodings[s])
					# the character in the run
					encoded_doc.append(encodings[count])
				index+=1
				

		encoded_doc = "".join(encoded_doc)
		return encoded_doc 

	def decompress(self, encoded_doc):
		assert len(encoded_doc)%32 == 0, "encoded_doc not in correct form"
		orig_grad = []
		"""Decompress by filling empty slots with zeros and reshape back using the original shape"""
		chunks = [encoded_doc[i:i+16] for i in range(0,len(encoded_doc),16)]
		for chunk in range(0,len(chunks),2):
			val = int(chunks[chunk], 2)
			for i in range(int(chunks[chunk + 1],2)):
				orig_grad.append(val)
		return orig_grad


	def reverse(self, s) :
		r = ""
		for i in range (len(s) - 1, -1, -1):
			r += s[i]
		return r


	def get_16_bit_binary(self, num):
		assert num < 2**16 , "Number out of range for RLE"
		s = ""
		while num > 0 :
			if (num & (1) == 1) :
				s += "1"
			else :
				s += "0"
			num = num >> 1
		while(len(s) < 16):
			s += "0"

		s =	reverse(s)
		return s

