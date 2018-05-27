// function dsigmoid(y){
// 	return y*(1-y);
// }
// let inputs = new Matrix(2,2);

// inputs.randomize();
// inputs.print();
// let outputs = Matrix.map(inputs,dsigmoid);
// outputs.print();
// console.log(outputs);
// let targets = [1,-1];
let res = new Matrix(1,1);
res.print();
let training_data =[
	{
		inputs:[0,1],
		targets:[1]
	},
	{
		inputs:[1,0],
		targets:[1]
	},
	{
		inputs:[0,0],
		targets:[0]
	},
	{
		inputs:[1,1],
		targets:[0]
	}
];

// console.log(training_data);

let nn = new NeuralNetwork(2,4,1);
let x = 10000;
console.log(x);
for(let i = 0; i < x; i++){

	for(data of training_data){
		nn.train(data.inputs,data.targets);
	}
}

console.log(nn.feedForward([1,0]));
console.log(nn.feedForward([0,1]));
console.log(nn.feedForward([0,0]));
console.log(nn.feedForward([1,1]));

// let output = nn.feedForward(input);


// console.log(output);
