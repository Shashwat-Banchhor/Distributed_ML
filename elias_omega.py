# Author: Shashwat Banchhor
# Email: shashwatbanchhor12@gmail.com
# Might need some alterations to be able to run with the grace framework

import torch

from grace_dl.dist import Compressor

from prefix_codes import omega_coding


class EliasOmega(Compress):
#     """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

	def __init__(self):
		super().__init__()
		self.encoded_document = []
		self.encoding = {}
		self.reverse_encoding ={}
		self.code_length = 0		

	def compress(self, grads):
		"""Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""
		encoded_doc = []
		
		index = 0
		s = -10000
		
		frequency = {}
		encodings = {}
		reverse_encodings = {}
		for gradient_idx in range(len(grads)):
			if gradient_idx==0:
				s = int(grads[gradient_idx])
				count = 1

			else :
				if (s!= int(grads[gradient_idx])):
					
					if(s<0):
						encoded_doc.append(omega_coding(2*(-s)))
						encodings[2*(-s)] = omega_coding(2*(-s))
						reverse_encodings[omega_coding(2*(-s))] = 2*(-s)
						# encoded_doc.append(val)
					else:
						encoded_doc.append(omega_coding(2*s+1))
						encodings[2*s+1] = omega_coding(2*s+1)
						reverse_encodings[omega_coding(2*s+1)] = 2*s+1
						# encoded_doc.append(val)
					encoded_doc.append(omega_coding(count))
					encodings[count] = omega_coding(count)
					reverse_encodings[omega_coding(count)] = count
					# encoded_doc.append(val)
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
				
				
				if(s<0):
						encoded_doc.append(omega_coding(2*(-s)))
						encodings[2*(-s)] = omega_coding(2*(-s))
						reverse_encodings[omega_coding(2*(-s))] = 2*(-s)
						# encoded_doc.append(val)
				else:
						encoded_doc.append(omega_coding(2*s+1))
						encodings[2*s+1] = omega_coding(2*s+1)
						reverse_encodings[omega_coding(2*s+1)] = 2*s+1
						# encoded_doc.append(val)
				encoded_doc.append(omega_coding(count))
				encodings[count] = omega_coding(count)
				reverse_encodings[omega_coding(count)] = count
				# encoded_doc.append(val)
				index+=1
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1


		encoded_doc = "".join(encoded_doc)


		self.encoded_document = encoded_doc
		self.encoding = encodings
		self.reverse_encoding = reverse_encodings
		self.code_length = len(encoded_doc)
		return encoded_doc


	def decompress(self):
		decoded_doc = []


		# run codeword
		s = ""

		# traverse the document bit-by-bit and check if current codeword is a valid codeword if yes initialize the run codeword to ""
		# alternately decode symbol and its count

		is_symbol = 1
		symbol = -1
		for char in self.encoded_document:
			s += char
			try:
				val = self.reverse_encoding[s]
				if is_symbol:
					if (val%2==0):
						symbol = -val/2
					else:
						symbol = int(((val-1)/2) + 0.5)
				else:
					repetitions = val
					for i in range(val):
						decoded_doc.append(symbol)
				s=""
			except:
				continue

			is_symbol = 1 - is_symbol

		print(decoded_doc)
		return decoded_doc


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


