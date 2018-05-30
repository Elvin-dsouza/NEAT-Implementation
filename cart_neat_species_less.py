import gym
import random
import copy
import numpy as np
import scipy.special
from statistics import mean, median
from collections import Counter
import math
import sys



# Hyper Parameters
# Alter based on the generation size

# mutation rates for various different types of mutations
# rate that a node gets added splitting an existing connection
mutation_add_rate = 0.3
# rate that a link gets added
mutation_link_rate = 0.05
# rate of altering a links weights by a gaussian distribution
mutation_weight_rate = 0.8
mutation_refresh_rate = 0.1


# chance of breeding happening over just mutation which is 1 - crossover_chance
crossover_rate = 0.75


perturbance_rate = 0.1

# interspecies mating rate
mating_rate = 0.001

# threshold value for distance between two species
species_threshold = 3.0

# constants to decide on which component affects compatibility
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

# TODO: implement relu instead of sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


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
		self.visited = []
		self.species_no = species_no
		self.genes = []
		self.fitness = 0.0
		self.nodes = []
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		for x in range(num_inputs):
			self.nodes.append(Node(self.nno,0))
			self.nno += 1
		for x in range(num_outputs):
			self.nodes.append(Node(self.nno,2))
			# for node in self.nodes:
			# 	if node.ntype == 0:
			# 		Genome.innovation+= 1
			# 		self.genes.append(Gene(node.node_number,self.nno,random.uniform(0,1),1,Genome.innovation))
			# print("current innovation_number" + str(local_innovation))
			self.nno += 1
		pass

	
	
	def split_nodes(self, nodea, nodeb):
		for gene in self.genes:
			if gene.input_node == nodea.node_number:
				if gene.output_node == nodeb.node_number:
					gene.enabled = 0
					node = Node(self.nno, 1)
					Genome.innovation += 1
					split_gene_a = Gene(gene.input_node, self.nno, 1, 1, Genome.innovation)
					Genome.innovation += 1
					split_gene_b = Gene(self.nno, gene.output_node,
										gene.weight, 1, Genome.innovation)
					self.genes.append(split_gene_a)
					self.genes.append(split_gene_b)
					self.nodes.append(node)
					self.nno += 1


	def join_nodes(self, nodea, nodeb):
		
		if nodea.connected(nodeb, self.genes):
			return False
		Genome.innovation +=1
		joint = Gene(nodea.node_number, nodeb.node_number,
		             random.uniform(0, 1), 1, Genome.innovation)
		self.genes.append(joint)
		self.visited = []
		if self.getPrevious(nodeb) == True:
			self.genes.pop()
			Genome.innovation -=1 
			return False
		else:
			return True
		

	def testSet1(self):
		self.split_nodes(self.nodes[0], self.nodes[4])
		self.split_nodes(self.nodes[1], self.nodes[4])
		self.split_nodes(self.nodes[0], self.nodes[6])
		Genome.innovation += 1
		self.genes.append(Gene(7, 8, random.uniform(0, 1), 1, Genome.innovation))
		# Genome.innovation += 1
		# self.genes.append(Gene(8, 9, random.uniform(0, 1), 1, Genome.innovation))

	
	def getPrevious(self, node):
		if(node.ntype == 0):
			return False
		if len(self.visited) > 0:
			for x in self.visited:
				if x == node.node_number:
					return True	

		self.visited.append(node.node_number)
		output = False
		for gene in self.genes:
			if gene.enabled == 1:
				if node.node_number == gene.output_node:
					output = self.getPrevious(self.nodes[gene.input_node-1])
		return output

	def feedforward(self,node):
		connections = []
		output = 0
		if node.ntype == 0:
			return node.output_value

		for gene in self.genes:
			if gene.output_node == node.node_number:
				connections.append([gene.input_node - 1, gene.weight])

		if len(connections) == 0:
			return 0
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
		# print("predict")
		for inode in self.nodes:
			if inode.ntype == 0:
				inode.output_value = inputs[inode.node_number-1]
		outputs = []
		for onode in self.nodes:
			if onode.ntype == 2:
				# print(onode.node_number)
				# input("output")
				outputs.append(self.feedforward(onode))
		return outputs

	def mutate(self):
		for gene in self.genes:
			if random.random() < perturbance_rate:
				gene.weight += random.gauss(0,perturbance_rate)
			#TODO:T 10% chance of assigned a new random value
		pass

	def random_node(self , ipIncluded = False,  opIncluded = False):
		pos = -1
		pos = random.randrange(0, len(self.nodes))
		node = self.nodes[pos]
		if ipIncluded == False and opIncluded == False: # only hidden
			while node.ntype == 0 or node.ntype == 2:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		elif ipIncluded == True and opIncluded == False:  # all excluding output
			# input("outside loop"+str(node.node_number))
			while node.ntype == 2:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		elif ipIncluded == False and opIncluded == True:  # all excluding input
			while node.ntype == 0:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		return pos


	def mutate_topology(self):
		# add node mutation
		if random.random() < mutation_add_rate:
				nodea = self.nodes[self.random_node(True, False)] # the first node can be an input but not an output
				nodeb = self.nodes[self.random_node(False, True)] # the second node can be an output but not an input
				self.split_nodes(nodea,nodeb) #split the two nodes this method auto creates the genes

		if random.random() < mutation_link_rate:
			# if self.nno != (self.num_inputs + self.num_outputs + 1): 
				# the first node can be an input but not an output
				nodea = self.nodes[self.random_node(True, False)]
				# the second node can be an output but not an input
				nodeb = self.nodes[self.random_node(False, True)]
				result = self.join_nodes(nodea, nodeb)
				counter = 0
				while result == False:
					if counter < self.nno * 2:
						nodea = self.nodes[self.random_node(True, False)]
						nodeb = self.nodes[self.random_node(False, True)]
						result = self.join_nodes(nodea, nodeb)
						counter+=1
						# print(counter)
					else:
						break
				# input("x")
					


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

		if len(self.genes) == 0 or len(xgenes) == 0:
			return 0

		if len(self.genes) > len(xgenes):
			
			N = len(self.genes)
			bgenes = self.genes
			sgenes = xgenes
			# print("\nBigger Genome: A"+str(N)+"\n")
		else:
			# print("\nBigger Genome: B\n")
			N = len(xgenes)
			bgenes = xgenes
			sgenes = self.genes
			# print("\nBigger Genome: B"+str(N)+"\n")
		
	
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
		# print("Excess: ", str(num_excess), " Disjoint ", str(num_disjoint), "Mean Weight Difference: ", str(mean(weight_differences)))
		sdelta = excess_component + disjoint_component + weight_component
		return sdelta

	def sortGenes(self):
		for passnum in range(len(self.genes)-1, 0, -1):
			for i in range(passnum):
				if self.genes[i].innovation_number > self.genes[i+1].innovation_number:
					temp = self.genes[i]
					self.genes[i] = self.genes[i+1]
					self.genes[i+1] = temp
		pass


	@staticmethod
	def same(agenes, bgenes):
		outgenes = []
		for i, ag in enumerate(agenes):
			for j, bg in enumerate(bgenes):
				if(i!=j):
					if ag.innovation_number == bg.innovation_number:
						choice = random.randrange(0, 2)
						if choice == 1:
							outgenes.append(ag)
						elif choice == 0:
							outgenes.append(bg)
		return outgenes
						


	@staticmethod
	def disjoint(agenes,bgenes):
		outgenes = []
		for i, ag in enumerate(agenes):
			for j, bg in enumerate(bgenes):
				if(i != j):
					if ag.innovation_number == bg.innovation_number:
						choice = random.randrange(0, 2)
						if choice == 1:
							outgenes.append(ag)
						elif choice == 0:
							outgenes.append(bg)
						agenes.pop(i)
		for ag in agenes:
			outgenes.append(ag)
		return outgenes
		
	@staticmethod
	def crossover(parentA, parentB):
		# assuming parent A is the fitter parent
		child = copy.deepcopy(parentA)
		child.visited = []
		child.genes = []
		# samegenes = Genome.same(parentA.genes, parentB.genes)
		dgenes = Genome.disjoint(parentA.genes, parentB.genes)
		child.genes.extend(dgenes)
		child.display_gene()
		child.sortGenes()
		# print("Parent A")
		# parentA.display_gene()
		# print("Parent B")
		# parentB.display_gene()
		# print("Child")
		# child.display_gene()
		# input("crossover")
		return child


	def display(self):
		for node in self.nodes:
			node.display()
		for gene in self.genes:
			gene.display()
	
	def display_gene(self):
		print("nodes")
		for node in self.nodes:
			print(str(node.node_number)+"\t", end="")
		print("\n\n")
		for gene in self.genes:
			print(str(gene.innovation_number)+"\t", end="")
		print("\n")
		for gene in self.genes:
			print(str(gene.input_node)+ " -> "+ str(gene.output_node)+"\t", end="")
		print("\n")


class Agent:
	def __init__(self, input_shape, output_shape, genome = False):
		self.score = 0
		self.fitness = 0.0
		self.global_fitness = 0.0
		if not genome:
			self.genome = Genome(input_shape, output_shape)
		else:
			self.genome = copy.deepcopy(genome)
		pass

class Species:
	speciesNumber = 0
	def __init__(self):
		self.organisms = []
		self.population = 0
		Species.speciesNumber += 1
		self.number = Species.speciesNumber
		self.repFitness = 0.0
		pass
	
	def add(self, agent):
		self.organisms.append(agent)
		self.population+=1
	
	def sort(self):
		for passnum in range(len(self.organisms)-1, 0, -1):
			for i in range(passnum):
				if self.organisms[i].fitness < self.organisms[i+1].fitness:
					temp = self.organisms[i]
					self.organisms[i] = self.organisms[i+1]
					self.organisms[i+1] = temp
		pass
	
	def removeUnfit(self):
		
		if len(self.organisms) > 5:
			self.sort()
			# print("Called")
			unfit = int(len(self.organisms)/2)
			# print(unfit)
			self.organisms = self.organisms[:-unfit]
	
	def reduce(self):
		temp =  self.organisms[0]
		self.organisms = []
		self.organisms.append(temp)
		# print(len(self.organisms))
		# input("x")
		self.population = 1

	@staticmethod
	def share(a,b):
		
		d = a.calculateSpeciesDelta(b.genes)
		if d > species_threshold:
			return 0
		else:
			return 1

	def print(self):
		print("---------------------------------------------------")
		print("Species Number: ", str(self.speciesNumber))
		print("Fitness: ", str(self.repFitness))
		print("Number of organisms (Population): ", len(self.organisms))
		print("---------------------------------------------------")

	
	



def test():
	a = Agent(4,2)
	b = Agent(4, 2)
	for _ in range(2):
		a.genome.mutate()
		b.genome.mutate()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate()
		b.genome.mutate()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		a.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		b.genome.mutate_topology()
		a.genome.mutate_topology()
		a.genome.mutate_topology()
		
	print("\nGenome A\n")
	a.genome.display_gene()
	print("\nGenome B\n")
	b.genome.display_gene()
	child =Genome.crossover(a.genome, b.genome)
	print("\nChild\n")
	child.display_gene()

# test()



next_generation = []
prev_generation = []
def calculate_fitness():
	# find each agents fitness values
	pass


def calculateAverageFitness(generation):
	for species in generation:
		total = 0
		for organism in species.organisms:
			total += organism.global_fitness
		species.repFitness = total/species.population


def totalAverageFitness(generation):
	total = 0
	for species in generation:
		total += species.repFitness
	return total

reproduction_count = 0
def speciate(org):
	global next_generation
	for x in next_generation:
		if Species.share(org.genome, x.organisms[0].genome, ) == 1:
			x.add(org)
			return x.number
	s = Species()
	s.add(org)
	next_generation.append(s)
	return s.number

def newGeneration():
	
	global reproduction_count
	reproduction_count = 0
	global next_generation
	children = []
	calculateAverageFitness(next_generation)
	totalFitness = totalAverageFitness(next_generation)
	for x in next_generation:
		x.sort()
		x.removeUnfit()
		breedCount = int(x.repFitness / totalFitness * gen_iterations) - 1
		for i in range(breedCount):
			
			if random.random() < crossover_rate:
				xx = random.randrange(0, len(x.organisms))
				xy = random.randrange(0, len(x.organisms))
				while xx == xy:
					xx = random.randrange(0, len(x.organisms))
					xy = random.randrange(0, len(x.organisms))

				if x.organisms[xy].global_fitness > x.organisms[xx].global_fitness:
						temp = xx
						xx = xy
						xy = temp
				childGenome = Genome.crossover(
                                    x.organisms[xx].genome, x.organisms[xy].genome)
				reproduction_count += 1
				# apply random chance of further mutation
				childGenome.mutate()
				childGenome.mutate_topology()
				childOrganism = Agent(i_shape, o_shape, childGenome)
				# TODO: optional check if it still belongs to the same species?
				# TODO: interspecies breeding
				# TODO: random enable disable genes
				children.append(childOrganism)
			else:
				xx = random.randrange(0, len(x.organisms))
				childGenome = copy.deepcopy(x.organisms[xx].genome)
				childGenome.mutate()
				childOrganism = Agent(i_shape, o_shape, childGenome)
				children.append(childOrganism)
				reproduction_count += 1
	# print(len(children))
	# input("children")
	for species in next_generation:
		species.reduce()
		

	for organism in children:
		speciate(organism)
	for species in next_generation:
		species.print()
	



def generateInitialPopulation():
	global next_generation
	next_generation = []
	s = Species()
	print("Initial pop: ", str(s.number))
	a = Agent(i_shape, o_shape)
	for _ in range(gen_iterations):
		b = copy.deepcopy(a)
		b.genome.mutate()
		# next_generation.append(b)
		s.add(b)
	next_generation.append(s)
	




def printGeneration(x):
	print("Current Generation", str(x))
	print("Number of Species: ", str(len(next_generation)))
	norg = 0
	
	for species in next_generation:
		print("species ", str(species.number), " Organisms: ",str(len(species.organisms)))
		print(species.repFitness)
		norg += len(species.organisms)
	print("Organisms: ", str(norg))
	print("Breed Count: ", str(reproduction_count))

def findFittest():
	global next_generation
	fittest = 0
	max_fitness = 0.0
	for species in next_generation:
		for organism in species.organisms:
			if organism.global_fitness > max_fitness:
				fittest = organism
	return fittest


def showFittest():
	organism = findFittest()
	prev_status = []
	score = 0
	prev_status = env.reset()
	for timestep in range(200):
		env.render()
		action = env.action_space.sample()
		out = organism.genome.predict(prev_status)
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
	print("Fittest Organism in Generation: score ", str(score))
	print("Number of nodes: ", str(len(organism.genome.nodes)))
	print("Number of Genes: ", str(len(organism.genome.genes)))
	organism.genome.display_gene()
	# input()
	pass
# generateInitialPopulation()
def run_generation(x,display=False):
	if x == 0:
		# first generation
		generateInitialPopulation()
		printGeneration(x)
	else:
		# other generations
		newGeneration()
		printGeneration(x)

	for i,species in enumerate(next_generation):
		for j,organism in enumerate(species.organisms):
			prev_status = []
			score = 0
			prev_status = env.reset()
			# input("Next Episode")
			for timestep in range(200):
				if display:
					env.render()
				action = env.action_space.sample()
				out = organism.genome.predict(prev_status)
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
			organism.global_fitness = score
	showFittest()
			# print(score)
		# current_agent.genome.display()
		# input("press Enter to continue")


# run_generation(1)
# run_generation(3)
# run_generation(4)
# run_generation(5)
# run_generation(6)
# run_generation(7)

for i in range(150):
	run_generation(i)
env.close()
