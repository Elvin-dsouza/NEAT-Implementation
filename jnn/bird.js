

class Bird{

	constructor(brain){
		this.y = height/2;
		this.x = 25;
		this.gravity = 0.4;
		this.velocity = 0;
		this.lift = -15;
		if(brain){
			this.brain = brain.copy();
		}
		else
		{
			this.brain  = new NeuralNetwork(5,4,2);
		}
		this.score = 0;
		this.fitness = 0;
		console.log("called");
	}

	show(){
		fill(255,100);
		image(img, this.x, this.y,25,25);
	}

	mutate(){
		this.brain.mutate(0.1);
	}

	update(){
		this.velocity += this.gravity;
		this.y += this.velocity;
		this.velocity*=0.9;
		// if(this.y > height){
		// 	this.y = height;
		// 	this.velocity = 0;
		// }
		// if(this.y < 0){
		// 	this.y = 0;
		// 	this.velocity = 0;
		// }
		this.score ++;
	}

	hitsBottom(){
		if(this.y > height || this.y < 0 ){
			return true;
		}
		else
			return false
	}

	think(pipes){
		let closest = null;
		let closestD = Infinity;
		for(let i = 0; i < pipes.length; i++)
		{
			let d = (pipes[i].x + pipes[i].w) - this.x;
			if(d < closestD && d > 0 )
			{
				closest = pipes[i];
				closestD = d;
			}
		}
		let inputs = [];
		inputs[0] = this.y/height;
		inputs[1] = 0;// this.velocity / 10;
		inputs[2] = closest.top/height;
		inputs[3] = closest.bottom/height;
		inputs[4] = closest.x/width;
		let output = this.brain.predict(inputs);
		if(output[0] > output[1])
			this.up();
	}
	up(){
		if(this.velocity > 0)
			this.velocity += this.lift;
	}
}
	
