
const NODE_TYPE_SENSOR = 0;
const NODE_TYPE_HIDDEN = 1;
const NODE_TYPE_OUTPUT = 2;

const SPECIES_THRESHOLD = 0.4;

const EXCESS_BIAS   = 0.4;
const DISJOINT_BIAS = 0.5;
const WEIGHT_BIAS   = 0.1;

const MUTATE_CONNECTION_RATE = 0.4;
const MUTATE_NODE_RATE = 0.4;

let innovation_number = 0;

class Genome{

	constructor(num_inputs, num_outputs){
		this.genes = []; //store all genes in this genome
		this.n = 0; // number of nodes in the genome incremented everytime a new node is added
		this.num_genes = 0; // incrememnted everytime a new gene is added
		this.nodes = []; // store all nodes in the neural net
		this.fitness = 0.0; // fitness for this particular genome
		for(let j = 0; j < num_inputs; j++)
			nodes[this.n] = new Node(NODE_TYPE_SENSOR);
			this.n++;
		for(let j = 0; j < num_outputs; j++)
			nodes[this.n] = new Node(NODE_TYPE_OUTPUT);
			this.n++;
	}

	structural_mutate(){

	}

	static numExcess(a,b){

	}
	static numDisjoint(a,b){

	}

	addNewNode(){
		let n = new Node(NODE_TYPE_HIDDEN);

	}
	addNewGene(input_node, output_node){
		innovation ++;
		gn = new Gene(input_node, output_node, random(-1,1), true, innovation);	
		this.genes[this.num_genes] = gn;
		this.num_genes++;
		return gn;
	}

	static disjoint(a,b){
		let N = 0;
		let na = numGenes(a);
		let nb = numGenes(b);

		if(na < 20 && nb < 20) // for normalisation 
			N = 1;
		else // N is the number of genes in the largest genome
			if(na > nb)
				N = na;
			else
				N = nb;
		let excess_comp = EXCESS_BIAS * numExcess(a,b) / N;
		let disjoint_comp = DISJOINT_BIAS * numDisjoint(a,b) / N;
		let delta = excess_comp + disjoint_comp + WEIGHT_BIAS * AvgWeightDelta(a,b);
		return delta;
	}

	share(){

	}

	static breed(a,b){
	
	}

}



class Node{
	constructor(nodeType){
		this.nodeType = nodeType;
		this.node_number = 0;

	}


}

class Gene{
	constructor(input, output, weight, enabled, innovation){
		this.input_node = input;
		this.output_node = output;
		this.weight = weight;
		this.enabled = enabled;
		this.innovation = innovation;
	}

	disableGene(){
		this.enabled = false;
	}
}