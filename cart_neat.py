import gym
import random
import copy
import numpy as np
import scipy.special
from statistics import mean, median
from collections import Counter
import math
# from graphics import *
import sys
sys.setrecursionlimit(1500)
mutation_add_rate = 0.03
mutation_link_rate = 0.05
mutation_weight_rate = 0.8
mutation_refresh_rate = 0.1


perturbance_rate = 0.1
mating_rate = 0.001

species_threshold = 3.0

coeff_excess = 1.0
coeff_disjoint = 1.0
coeff_weights = 0.4

gen_iterations = 150
gen_timesteps = 200
generation = []

i_shape = 4
o_shape = 2

env = gym.make('CartPole-v1')


current_innovation = 0


def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def trace(nodea, nodeb, genes):

class Gene:
	
	def __init__(self, input_node, output_node ,weight, enabled,innovation_number):
		self.input_node = input_node
		self.output_node = output_node
		self.weight = weight
		self.enabled = enabled
		self.innovation_number = innovation_number
		pass


	def display(self):
		print("-----------------------------------")
		print("Gene | Innovation Number: ",str(self.innovation_number))
		print("In: ",str(self.input_node),"  Out: ", str(self.output_node))
		print("Weight: ",str(self.weight))
		print("enabled: ",str(self.enabled))

	def match(self, bgenes):
		if self.innovation_number == bgenes.innovation_number:
			return True
		else:
			return False

	
	# def isDisjoint(self, bgene):
	# 	if self.innovation_number != bgene.innovation_number:
	# 		return True
	# 	else:
	# 		return False

	def disjoint(self, genes):
		found = False
		for gene in genes:
			if self.innovation_number == gene.innovation_number:
				found = True
		if found:
			return False
		else:
			return True


class Node:
	def __init__(self, node_number, ntype):
		self.node_number = node_number
		self.ntype = ntype
		self.activated = 0
		self.output_value = 0.0
		pass


	def display(self):
		print("-----------------------------------")
		print("Node Number: ",str(self.node_number))
		print("Node Type: ",str(self.ntype),"  Activated: ", str(self.activated))
		print("Output Value: ",str(self.output_value))

		pass

	def getPrevious(genes)
		prev = []
		for gene in genes:
			if self.node_number == gene.output_node:
				prev.append(gene.input_node)


	


	def connected(self,b,genes):
		for gene in genes:
			if gene.input_node == b.node_number:
				if gene.output_node == self.node_number:
					return True

			if gene.output_node == b.node_number:
				if gene.input_node == self.node_number:
					return True


		return False

		


class Genome:
	innovation = 0
	def __init__(self, num_inputs, num_outputs, species_no = -1):
	
		self.nno = 1
		self.species_no = species_no
		self.genes = []
		self.fitness = 0.0
		self.nodes = []
		for x in range(num_inputs):
			self.nodes.append(Node(self.nno,0))
			self.nno += 1
		for x in range(num_outputs):
			self.nodes.append(Node(self.nno,2))
			for node in self.nodes:
				if node.ntype == 0:
					Genome.innovation+= 1
					self.genes.append(Gene(node.node_number,self.nno,random.uniform(0,1),1,Genome.innovation))
			# print("current innovation_number" + str(local_innovation))
			self.nno += 1
		pass

	def activate(self, node):
		inputs = []
		# print("called")
		local_sum = 0
		if node.activated == 0:
			if node.ntype != 2:
				for gene in self.genes:
					if node.node_number == gene.output_node:
						if gene.enabled == True:
							if self.nodes[gene.input_node-1].activated == 0:
								self.activate(self.nodes[gene.input_node-1])
								# input("Activated 1..")
								# print(str(gene.weight), "*",str(self.nodes[gene.input_node-1].output_value))
								local_sum += gene.weight * self.nodes[gene.input_node-1].output_value
								node.activated = 1
							else:
								# print(str(gene.weight), "*",str(self.nodes[gene.input_node-1].output_value))
								local_sum += gene.weight * self.nodes[gene.input_node-1].output_value
			node.output_value = sigmoid(local_sum)	
		else:
			return
	
	def feedforward(self,node):
		connections = []
		output = 0
		if node.ntype == 0:
			return node.output_value

		for gene in self.genes:
			if gene.output_node == node.node_number:
				connections.append([gene.input_node - 1, gene.weight])

		if len(connections) == 0:
			return 
		else:
			for link in connections:
				out = 0
				if self.nodes[link[0]].ntype == 0:
					out = self.nodes[link[0]].output_value
				else:
					out = self.feedforward(self.nodes[link[0]])
				output += (out * link[1])
			return sigmoid(output)

	def predict(self, inputs):
		print("predict")
		for inode in self.nodes:
			if inode.ntype == 0:
				inode.output_value = inputs[inode.node_number-1]
		outputs = []
		for onode in self.nodes:
			if onode.ntype == 2:
				print(onode.node_number)
				# input("output")
				outputs.append(self.feedforward(onode))
		return outputs


	def propogate(self, inputs):
		# set all the input nodes
		for inode in self.nodes:
			if inode.ntype == 0:
				# print(inode.node_number)
				inode.output_value = inputs[inode.node_number-1]

		# calculate all the hidden nodes values
		for node in self.nodes:
			if node.ntype == 1:
				self.activate(node)
		outputs = []
		# find out the outputs
		for onode in self.nodes:
			if onode.ntype == 2:
				output_sum = 0;
				for gene in self.genes:
					if gene.enabled == 1:
						if gene.output_node == onode.node_number:
							if self.nodes[gene.input_node-1].activated == 0:
								self.activate(self.nodes[gene.input_node-1])
								output_sum = gene.weight * self.nodes[gene.input_node-1].output_value
							else:
								output_sum = gene.weight * self.nodes[gene.input_node-1].output_value
				outputs.append(sigmoid(output_sum))

		for node in self.nodes:
			node.activated = 0
		return outputs



	def mutate(self):
		for gene in self.genes:
			if random.random() < mutation_weight_rate:
				gene.weight += random.gauss(0,perturbance_rate)
			# 10% chance of assigned a new random value
		pass

	def mutate_topology(self):
		# add node mutation
		for gene in self.genes:
			if random.random() < mutation_add_rate:
				gene.enabled = 0
				node = Node(self.nno,1)
				Genome.innovation += 1
				split_gene_a = Gene(gene.input_node, self.nno,1,1,Genome.innovation)
				Genome.innovation += 1
				split_gene_b = Gene(self.nno,gene.output_node,gene.weight,1,Genome.innovation)
				self.genes.append(split_gene_a)
				self.genes.append(split_gene_b)
				self.nodes.append(node)
				self.nno+=1

		# add link mutation
		# loop through all the nodes
		for node in self.nodes:
			# for each node check if its connected to another node
			for checknode in self.nodes:
				if checknode.node_number != node.node_number:
					if checknode.ntype == 1:
						if node.ntype != 0:
							if not node.connected(checknode,self.genes):
								if random.random() < mutation_link_rate:
									print("mtate"+str(node.node_number)+" "+str(checknode.node_number))
									Genome.innovation +=1
									link_gene = Gene(checknode.node_number,node.node_number,random.uniform(0,1),1,Genome.innovation)
									self.genes.append(link_gene)
					elif checknode.ntype == 0:
						if node.ntype != 0:
							if not node.connected(checknode,self.genes):
								if random.random() < mutation_link_rate:
									print("mtate"+str(node.node_number)+" "+str(checknode.node_number))
									Genome.innovation +=1
									link_gene = Gene(checknode.node_number,node.node_number,random.uniform(0,1),1,Genome.innovation)
									self.genes.append(link_gene)

					

				# if one node is not connected flag it
				# loop through the whole connections list checking for a connection between the nodes
				# if no connection exists create a connection between the two
				# if a connection exists reset the flag to -1
				# when a link has been established break
				# if the node being checked is a hidden node it can only be linked with other hidden or output nodes
				# if the node being checked is an input node it can only be linked with a hidden or output node
	# def displayNetwork(self):

	# 	win = GraphWin("Neural Network", 500, 500)
	# 	c = Circle(Point(50,50), 10)
	# 	c.draw(win)
	# 	win.getMouse() # pause for click in window
	# 	win.close()
	# 	pass
				


					
	def calculateSpeciesDelta(self, xgenes):
		bgenes = []
		sgenes = []
		weight_differences = []
		num_excess = 0
		num_disjoint = 0
		excess_component = 0
		disjoint_component = 0
		weight_component = 0
		sdelta = 0.0
		N = 0
		if len(self.genes) > len(xgenes):
			
			N = len(self.genes)
			bgenes = self.genes
			sgenes = xgenes
			print("\nBigger Genome: A"+str(N)+"\n")

		else:
			# print("\nBigger Genome: B\n")
			N = len(xgenes)
			bgenes = xgenes
			sgenes = self.genes
			print("\nBigger Genome: B"+str(N)+"\n")
		
	
		for bgene in bgenes:
			for sgene in sgenes:
				if bgene.innovation_number == sgene.innovation_number:
					weight_differences.append(abs(bgene.weight - sgene.weight))
		last = sgenes[len(sgenes)-1].innovation_number	
		for bgene in bgenes:
			if bgene.innovation_number > last:
				num_excess += 1

		for bgene in bgenes:
			if bgene.disjoint(sgenes):
				if bgene.innovation_number < last:
					num_disjoint += 1

		last = bgenes[len(bgenes)-1].innovation_number		
		for sgene in sgenes:
			if sgene.disjoint(bgenes):
				if sgene.innovation_number < last:
					num_disjoint += 1

		for sgene in sgenes:
			if sgene.innovation_number > last:
				num_excess += 1

		excess_component = coeff_excess * num_excess/ N
		disjoint_component = coeff_disjoint * num_disjoint / N
		weight_component = coeff_weights * mean(weight_differences)
		print("Excess: ", str(num_excess), " Disjoint ", str(num_disjoint), "Mean Weight Difference: ", str(mean(weight_differences)))
		sdelta = excess_component + disjoint_component + weight_component
		return sdelta




	def display(self):
		for node in self.nodes:
			node.display()
		for gene in self.genes:
			gene.display()


class Agent:
	def __init__(self, input_shape, output_shape, genome = False):
		self.score = 0
		if not genome:
			self.genome = Genome(input_shape, output_shape)
		else:
			self.genome = genome
		pass

	

def test():
	# b = Agent(2,2)
	# a = copy.deepcopy(b)
	# # a.genome.displayNetwork()
	# b.genome.mutate()
	# a.genome.mutate()
	# a.genome.mutate_topology()
	# a.genome.mutate_topology()
	# # b.genome.mutate()
	# b.genome.mutate_topology()
	# b.genome.mutate_topology()
	# b.genome.mutate_topology()
	# b.genome.mutate_topology()
	# # b.genome.mutate_topology()
	# # b.genome.mutate_topology()
	# print("\n\n\nGenome A\n\n\n")
	# a.genome.display()
	# print("\n\n\nGenome B\n\n\n")
	# b.genome.display()
	# delta = a.genome.calculateSpeciesDelta(b.genome.genes)
	# # print(len(b.genome.genes))
	# print(delta)
	a = Agent(4,2)
	a.genome.display()
	out = a.genome.predict([0.1,0.2,0.3,0.4])
	input("press enter to continue....")
	for _ in range(10):
		a.genome.mutate()
		a.genome.mutate_topology()
		a.genome.display()
		out = a.genome.predict([0.1,0.2,0.3,0.4])
		input("press enter to continue....")
	print(out)





test()

agents =[]
def run_generation(x):
	global agents
	if x == 0:
		a = Agent(4,2)
		for i in range(gen_iterations):
			x = copy.deepcopy(a)
			x.genome.mutate()
			x.genome.mutate_topology()
			x.genome.mutate_topology()
			x.genome.mutate_topology()

			agents.append(x)
	for episode in range(gen_iterations):
		prev_status = []
		score = 0
		prev_status = env.reset()
		current_agent = agents[episode]
		input("x")
		for timestep in range(200):
			# env.render()
			action = env.action_space.sample()
			out = current_agent.genome.predict(prev_status)
			# print(out)
			if(out[0] > out[1]):
				action = 0
			else:
				action = 1
			status, reward, done, info = env.step(action)
			score += reward
			prev_status = status
			if done:
				break
		current_agent.score = score
		print(score)
		current_agent.genome.display()
		# input("press Enter to continue")


# run_generation(0)
env.close()
