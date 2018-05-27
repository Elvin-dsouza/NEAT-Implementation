

let nn;

let training_data = [
	{
		inputs:[1,0],
		outputs:[1]
	},
	{
		inputs:[0,1],
		outputs:[1]
	},
	{
		inputs:[0,0],
		outputs:[0]
	},
	{
		inputs:[1,1],
		outputs:[0]
	}
];
function leaky_relu(x){
	if(x > 0)
		return x;
	else
		return (0.01 * x);
}

function setup(){
	console.log("setup")
	createCanvas(400,400);
	nn = new NeuralNetwork();
	let layer  = nn.add_layer(2,INPUT_LAYER);
	 	layer = nn.add_layer(5,HIDDEN_LAYER,leaky_relu,layer);
	    layer = nn.add_layer(5,HIDDEN_LAYER,leaky_relu,layer);
	 	layer = nn.add_layer(1,OUTPUT_LAYER,leaky_relu,layer);
}


function draw(){
	background(0);
	for(let i=0; i < 100; i++){
		let data = random(training_data)
		nn.train(data.inputs,data.outputs);
		if(i == 100)
			console.log("training complete");
	}
	let resolution = 10;
	let cols = width/resolution;
	let rows = height/resolution;
	for(let i = 0; i<cols; i++){
		for(let j=0; j< rows; j++){
			let x1 = i /cols;
			let x2 = j /rows;
			let inputs=[x1,x2];
			let y = nn.predict(inputs);
			noStroke();
			fill(y*255);
			rect(i*resolution,j*resolution,resolution,resolution);
		}
	}

}