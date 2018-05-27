function drawnn(Bbird){
	var copy = Bbird.brain;
	let minx = 200 + 20;
	let miny = 20  + 5;
	inputs = [];
	hidden = [];
	outputs= [];

	let num_inputs = copy.input_nodes;
	let num_hidden = copy.hidden_nodes;
	let num_output = copy.output_nodes;

	fill(255,100);
	rect(200,10,190,100);

	//draw inputs
	fill('#ff0000');
	let iter_y = miny;
	for(let i = 0; i < num_inputs; i++)
	{
		stroke(0);
		let obj = new  Object();
		obj.x = minx;
		obj.y = iter_y;
		ellipse(obj.x
			, obj.y,5,5);
		iter_y = i * 10 + miny;
		inputs.push(obj);

	}

	iter_y = miny;
	for(let i = 0; i <num_hidden; i++)
	{
		stroke(0)
		let obj = new  Object();
		obj.x = minx + 30;
		obj.y = iter_y;
		ellipse(obj.x, obj.y,5,5);
		iter_y = i * 10 + miny;
		hidden.push(obj);
		for(let j = 0; j < inputs.length; j++)
		{
		
			
					
			if(copy.weights_ih.data[i][j] > 0.5){
				stroke('#00FF00');
				line(inputs[j].x, inputs[j].y, hidden[i].x, hidden[i].y);

			}
			if(copy.weights_ih.data[i][j] < 0){
				stroke('#00FF00');
				line(inputs[j].x, inputs[j].y, hidden[i].x, hidden[i].y);

			}
			else
			{
				stroke(150);
				line(inputs[j].x, inputs[j].y, hidden[i].x, hidden[i].y);

			}
		}

	}

	iter_y = miny+40;
	for(let i = 0; i <num_output; i++)
	{
		
		let obj = new  Object();
		obj.x = minx + 70;
		obj.y = iter_y;
		ellipse(obj.x, obj.y,5,5);
		iter_y = i * 10 + miny;
		outputs.push(obj);
		for(let j = 0; j < hidden.length; j++)
		{
			if(copy.weights_ho.data[i][j] > 0.5){
				stroke('#00FF00');
				line(hidden[j].x, hidden[j].y, outputs[i].x, outputs[i].y);
			}
			else if(copy.weights_ho.data[i][j] < 0){
				stroke(150);
				line(hidden[j].x, hidden[j].y, outputs[i].x, outputs[i].y);
			}
			else
			{
				stroke('#FF0000');
				line(hidden[j].x, hidden[j].y, outputs[i].x, outputs[i].y);

			}
		}

	}

}


