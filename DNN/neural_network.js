//activation functions


//standard logistic sigmoid function
function sigmoid(x){
	return 1/(1 + Math.exp(-x));
}

function dsigmoid(y){
	return y * (1- y);
}


//leaky rectified linear units function
function leaky_relu(x){
	if(x > 0)
		return x;
	else
		return (0.01 * x);
}

function dleaky_relu(y){
	if(y > 0)
		return 1;
	else
		return (0.01);
}
const INPUT_LAYER = 0;
const HIDDEN_LAYER = 1;
const OUTPUT_LAYER = 2;

class NeuralNetwork{

	constructor(x){
		this.layers = [];
		if(x instanceof NeuralNetwork)
		{
			for(let i=0; i < x.num_layers; i++)
					this.layers.push(x.layers[i].copy());
			this.num_layers = x.num_layers;
			this.num_outputs = x.num_outputs;
			this.num_inputs = x.num_inputs;
			this.learning_rate = x.learning_rate;
		}
		// instantiate an array of layers
		this.num_layers = 0;
		this.num_outputs = 0;
		this.num_inputs = 0;
		this.learning_rate = 0.01;
	}

	copy(){
		// console.log(new NeuralNetwork(this));
		return new NeuralNetwork(this);
	}

	add_layer(num_nodes, type, activation_function, prev_layer){
		// get the number of nodes from the previous layer
		let prev_nodes = 0;
		if(prev_layer)
		{
			prev_nodes = prev_layer.getNodes();
		}
		let layer;
		if(type === INPUT_LAYER){ 
			this.num_inputs = num_nodes;
			//input nodes have no activation or previous layer before it
			layer = new Layer(num_nodes, type);
		}
		else
		{
			layer = new Layer(num_nodes, type, activation_function, prev_nodes);
			if(type === OUTPUT_LAYER)
				this.num_outputs = num_nodes;
		}
		this.num_layers++;
		//push the new layer to the array
		this.layers.push(layer);
		return layer; //return the layer just incase
	}


	predict(input_array){
		//get inputs and put them into a matrix wrapper
		let inputs = Matrix.fromArray(input_array);
		let temp_mat = inputs;
		for(let i = 1; i < this.layers.length; i++){
			temp_mat = Matrix.multiply(this.layers[i].weights,temp_mat);
			temp_mat.add(this.layers[i].bias);
			temp_mat.map(this.layers[i].activation_function);
		}
		return temp_mat.toArray();
	}

	mutate(rate){
		function gaussianMutation(val){
			if(Math.random() < rate){
				// console.log("mutated");
				return val + randomGaussian(0,0.1);

			}
			else
				return val;
		}
		for(let i=1; i < this.layers.length; i++)
		{
			this.layers[i].weights.map(gaussianMutation);
			this.layers[i].bias.map(gaussianMutation);
		}
	}

	train(input_array, target_array)
	{
		let inputs = Matrix.fromArray(input_array);
		let targets = Matrix.fromArray(target_array);
		let layer_outputs = [];
		layer_outputs[0] = inputs;
		// console.log("outputs: "+layer_outputs[0]);
		for(let i = 1; i < this.layers.length; i++){
			// console.log(layer_outputs[i-1]);
			// let weights = this.layers[i].weights;
			layer_outputs[i] = Matrix.multiply(this.layers[i].weights,layer_outputs[i-1]);
			layer_outputs[i].add(this.layers[i].bias);
			layer_outputs[i].map(leaky_relu);

		}
		// console.table(layer_outputs);
		let layer_errors = [];
		layer_errors[this.layers.length - 1] =  Matrix.subtract(targets, layer_outputs[this.layers.length-1]);
		// print(layer_errors);
		for(let i=this.layers.length-1; i > 0 ; i--)
		{
			// console.log(i);
			let gradient = Matrix.map(layer_outputs[i], dleaky_relu);
			gradient.multiply(layer_errors[i]);
			gradient.multiply(this.learning_rate);

			// console.log("gradient"+i);
			// console.log(gradient);
			// console.log("errors");
			// console.log(layer_errors[i]);
			// console.log("outputs");
			// console.log(layer_outputs[i]);
			// console.table(gradient);
			let transpose_prev = Matrix.transpose(layer_outputs[i-1]);
			let delta = Matrix.multiply(gradient,transpose_prev);
			this.layers[i].weights.add(delta);
			this.layers[i].bias.add(gradient);
			if((i-1)!=0)
			{
				let weights_transpose = Matrix.transpose(this.layers[i].weights);
				layer_errors[i-1] = Matrix.multiply(weights_transpose,layer_errors[i]);
			}
		}
	
	}
}


class Layer{

	constructor(nodes, type, activation_function, prev_nodes){
		if(type === INPUT_LAYER){
			this.nodes = nodes;
			this.type = type;
		
		}
		else if(type === OUTPUT_LAYER || type === HIDDEN_LAYER)
		{
			this.nodes = nodes;
			this.type = type;
			this.prev_nodes = prev_nodes;
			this.activation_function = activation_function;
			this.bias = new Matrix(this.nodes,1);
			this.weights = new Matrix(this.nodes,this.prev_nodes);
			this.weights.randomize();
			this.bias.randomize();
			this.dfunc = dleaky_relu;
		}
		
	}

	copy(){
		if(this.type === INPUT_LAYER)
			return new Layer(this.nodes, this.type);
		else
			return new Layer(this.nodes, this.type, this.activation_function, this.prev_nodes);
	}

	getNodes()
	{
		return this.nodes;

	}
}