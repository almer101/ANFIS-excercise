import numpy as np
from math import cos
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class NeuroFuzzy:

	def __init__(self, rules_number):
		self.rules_number = rules_number

		# generate `rules_number` long arrays below
		self.As = self.generate_random(rules_number)
		self.Bs = self.generate_random(rules_number)
		self.Cs = self.generate_random(rules_number)
		self.Ds = self.generate_random(rules_number)
		self.ps = self.generate_random(rules_number)
		self.qs = self.generate_random(rules_number)
		self.rs = self.generate_random(rules_number)

		self.initialize_deltas(rules_number)
		self.count = 0
		self.gg = True

	def initialize_deltas(self, n):
		self.dA = np.zeros(n)
		self.dB = np.zeros(n)
		self.dC = np.zeros(n)
		self.dD = np.zeros(n)
		self.dp = np.zeros(n)
		self.dq = np.zeros(n)
		self.dr = np.zeros(n)

	def generate_random(self, n):
		value = []
		for i in range(n):
			value.append(random.random() * 2.0 - 1.0)

		return np.array(value)

	def evaluate(self, x, y):
		return self.forward((x,y), 0.0)

	def forward(self, inpt, y):
		# input is (x, y)
		self.last_input = inpt
		self.last_y = y
		self.A_out = []
		self.B_out = []

		x, y = inpt

		outputs = []

		for i in range(len(self.As)):
			a = self.As[i]
			b = self.Bs[i]
			c = self.Cs[i]
			d = self.Ds[i]

			# 1. layer
			o1 = 1.0 / (1.0 + np.exp(b * (x - a)))
			self.A_out.append(o1)
			o2 = 1.0 / (1.0 + np.exp(d * (y - c)))
			self.B_out.append(o2)

			# 2. layer
			outputs.append(o1 * o2)

		self.Ws = outputs

		# 3. layer
		outputs = np.array(outputs)
		suma = 0.0
		for o in outputs:
			suma += o

		self.w_sum = suma
		outputs /= suma

		# 4. layer
		fs = []
		fsum = 0.0
		for i, o in enumerate(outputs):
			value = self.ps[i] * x + self.qs[i] * y + self.rs[i]
			value *= o
			fsum += value
			fs.append(value)

		self.out = fsum
		self.fs = fs

		return fsum

	def backward(self):
		wifi_sum = 0.0
		self.count += 1
		for i in range(len(self.Ws)):
			wifi_sum += self.Ws[i] * self.fs[i]

		self.dp += -(self.last_y - self.out) * (self.Ws / self.w_sum) * self.last_input[0]
		self.dq += -(self.last_y - self.out) * (self.Ws / self.w_sum) * self.last_input[1]
		self.dr += -(self.last_y - self.out) * (self.Ws / self.w_sum)

		for i in range(len(self.dp)):
			self.dA[i] += -(self.last_y - self.out) * (self.fs[i] * self.w_sum - wifi_sum) / (self.w_sum**2) * self.B_out[i] * self.Bs[i] * self.A_out[i] * (1.0 - self.A_out[i])
			self.dB[i] += -(self.last_y - self.out) * (self.fs[i] * self.w_sum - wifi_sum) / (self.w_sum**2) * self.B_out[i] * self.A_out[i] * (self.A_out[i] - 1.0) * (self.last_input[0] - self.As[i])
			self.dC[i] += -(self.last_y - self.out) * (self.fs[i] * self.w_sum - wifi_sum) / (self.w_sum**2) * self.A_out[i] * self.Ds[i] * self.B_out[i] * (1.0 - self.B_out[i])
			self.dD[i] += -(self.last_y - self.out) * (self.fs[i] * self.w_sum - wifi_sum) / (self.w_sum**2) * self.A_out[i] * self.B_out[i] * (self.B_out[i] - 1.0) * (self.last_input[1] - self.Cs[i])

	def update_weights(self, eta = 1e-03):
		# average the corrections
		self.dA /= float(self.count)
		self.dB /= float(self.count)
		self.dC /= float(self.count)
		self.dD /= float(self.count)
		self.dp /= float(self.count)
		self.dq /= float(self.count)
		self.dr /= float(self.count)

		# update weights
		self.As -= eta * self.dA
		self.Bs -= eta * self.dB
		self.Cs -= eta * self.dC
		self.Ds -= eta * self.dD

		self.ps -= eta * self.dp
		self.qs -= eta * self.dq
		self.rs -= eta * self.dr

		self.count = 0
		self.initialize_deltas(self.rules_number)


def function_elementwise(X,Y, fun):
	result = []
	for i in range(len(X)):
		result.append([])
		for j in range(len(X[i])):
			x = X[i][j]
			y = Y[i][j]
			value = fun(x, y)
			result[i].append(value)
	return np.array(result)

def function(x,y):
	return ((x - 1.0)**2 + (y + 2.0)**2 - 5.0 * x * y + 3.0) * (cos(x / 5.0) ** 2)

def fit(training_data, training_labels, epochs, rules_number, optimization_type='stohastic'):
	pass

if __name__ == "__main__":

	training_data = []
	xs = [i for i in range(-4, 5)]
	ys = [i for i in range(-4, 5)]

	for i in range(len(xs)):
		for j in range(len(ys)):
			training_data.append((float(xs[i]), float(ys[j])))
	
	training_labels = []

	for x, y in training_data:
		f = ((x - 1.0)**2 + (y + 2.0)**2 - 5.0 * x * y + 3.0) * (cos(x / 5.0) ** 2)
		training_labels.append(f)


	optimization_type = 'stohastic' # stohastic batch

	net = NeuroFuzzy(rules_number = 8)
	outs = []
	
	for i in range(4000):
		error = 0.0
		outs = []
		for j in range(len(training_data)):
			out = net.forward(training_data[j], training_labels[j])
			outs.append(out)
			net.backward()

			error += (out - training_labels[j])**2
			
			if optimization_type == 'stohastic':
				net.update_weights(eta=1e-03)

		if optimization_type == 'batch':
			net.update_weights(eta=1e-02)

		if i % 100 == 0:
			print(f"Error for epoch {i} is {error / (2 * len(training_labels))}")

	ax = plt.axes(projection='3d')

	X = np.linspace(-4, 4, 50)
	Y = np.linspace(-4, 4, 50)
	# ax.scatter(X, Y, np.array(training_labels), 'green')
	# ax.scatter(X, Y, np.array(outs), 'blue')

	X, Y = np.meshgrid(X, Y)
	Z = function_elementwise(X, Y, function)
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='black')
	
	Z = function_elementwise(X, Y, net.evaluate)
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='red')

	# ===================================================================================================

	# ax = plt.axes(projection='3d')

	# X = np.linspace(-4, 4, 50)
	# Y = np.linspace(-4, 4, 50)
	# # ax.scatter(X, Y, np.array(training_labels), 'green')
	# # ax.scatter(X, Y, np.array(outs), 'blue')

	# X, Y = np.meshgrid(X, Y)
	# # Z = function_elementwise(X, Y, function)
	# # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='black')
	
	# Z = function_elementwise(X, Y, lambda a,b: net.evaluate(a,b) - function(a,b))
	# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='black')

	x_range = np.linspace(-4, 4, 100)
	plt.subplots(3, 2)

	for i in range(len(net.As)):
		plt.subplot(4, 2, i + 1)
		ys = []
		product = []
		y_range = []
		a = net.As[i]
		b = net.Bs[i]
		c = net.Cs[i]
		d = net.Ds[i]

		for j in range(len(x_range)):
			ys.append(1.0 / (1.0 + np.exp(b * (x_range[j] - a))))
			y_range.append(1.0 / (1.0 + np.exp(d * (x_range[j] - c))))
			product.append(ys[-1] * y_range[-1])

		plt.plot(x_range, np.array(ys), label='x', linestyle='dotted')
		plt.plot(x_range, np.array(y_range), linestyle='dashed', label='y')
		plt.plot(x_range, np.array(product), label='product')


	plt.legend()
	plt.show()

