// function sigmoid(x){
// 	return 1/(1 + Math.exp(-x));
// }

// function dsigmoid(y){
// 	return y * (1-y);
// }

//Leaky rectified linear units to prevent neuron death and increase effeciency of learning
function sigmoid(x){
	if(x > 0)
			return x;
	else
		return (0.01 * x);
}

function dsigmoid(y){
		if(y > 0)
		return 1;
	else
		return 0.01;
}

// function relu(x){
// 	if(x > 0)
// 			return x;
// 	else
// 		return (0.01 * x);
// }

// function drelu(y){
// 	if(y > 0)
// 		return 1;
// 	else
// 		return 0.01;
// }
class NeuralNetwork{
	constructor(x, y, z){
		if(x instanceof NeuralNetwork){
			this.input_nodes = x.input_nodes;
			this.output_nodes = x.output_nodes;
			this.hidden_nodes = x.hidden_nodes;
			this.weights_ih = x.weights_ih.copy();
			this.weights_ho = x.weights_ho.copy();
			this.bias_h = x.bias_h.copy();
			this.bias_o = x.bias_o.copy();
		}
		else
		{
			this.input_nodes = x;
			this.hidden_nodes = y;
			this.output_nodes = z;


			this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
			this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
			this.weights_ih.randomize();
			this.weights_ho.randomize();

			this.bias_h = new Matrix(this.hidden_nodes, 1);
			this.bias_o = new Matrix(this.output_nodes, 1);

			this.bias_h.randomize();
			this.bias_o.randomize();
		}
	}

	predict(input_array)
	{
		let inputs = Matrix.fromArray(input_array);
		let hidden = Matrix.multiply(this.weights_ih,inputs);
		hidden.add(this.bias_h);
		hidden.map(sigmoid);
		let outputs = Matrix.multiply(this.weights_ho,hidden);
		outputs.add(this.bias_o);
		outputs.map(sigmoid);
		return outputs.toArray();
	}


	train(input_array, target_array)
	{
		let inputs = Matrix.fromArray(input_array);
		let targets = Matrix.fromArray(target_array);

		let hidden = Matrix.multiply(this.weights_ih,inputs);
		hidden.add(this.bias_h);
		hidden.map(sigmoid);


		let outputs = Matrix.multiply(this.weights_ho,hidden);
		outputs.add(this.bias_o);
		outputs.map(sigmoid);

		output_errors = Matrix.subtract(targets,outputs);
		
		let output_gradients = Matrix.map(outputs,dsigmoid);
		output_gradients.multiply(output_errors);
		output_gradients.multiply(0.1);

		let hidden_transpose = Matrix.transpose(hidden);
		let ho_wd = Matrix.multiply(output_gradients,hidden_transpose);
		
		this.weights_ho.add(ho_wd);
		this.bias_o.add(output_gradients);

		let weight_ho_t = Matrix.transpose(this.weights_ho);
		let hidden_errors = Matrix.multiply(weight_ho_t,output_errors);

		let hidden_gradient = Matrix.map(hidden,dsigmoid);
		hidden_gradient.multiply(hidden_errors);
		hidden_gradient.multiply(0.1);

		let inputs_t = Matrix.transpose(inputs);
		let ih_wd = Matrix.multiply(hidden_gradient,inputs_t);

		this.weights_ih.add(ih_wd);
		this.bias_h.add(hidden_gradient);

	}

	serialize(){
		return JSON.stringify(this);
	}

	static deserialize(data){
		if(typeof data == 'string')
			data = JSON.parse(data)
		let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
		nn.weights_ih = Matrix.deserialize(data.weights_ih);
		nn.weights_ho =  Matrix.deserialize(data.weights_ho);
		nn.bias_o =  Matrix.deserialize(data.bias_o);
		nn.bias_h =  Matrix.deserialize(data.bias_h);
		return nn;
	}

	copy()
	{
		return new NeuralNetwork(this);
	}

	mutate(rate) {
	    function mutate(val){
	      if(Math.random() < rate){
	        // return 2*Math.random()-1;
	        console.log("mutated");
	        return val + randomGaussian(0,0.1);
	      }
	      else
	      {
	        return val;
	      }
	    }
	    this.weights_ih.map(mutate);
	    this.weights_ho.map(mutate);
	    this.bias_h.map(mutate);
	    this.bias_o.map(mutate);
  	}
}