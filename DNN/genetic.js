let genCount = 0;
let max_fitness = 0;
var bestBird;
function nextGeneration(){
	calcFitness();
	for(let i = 0; i < TOTAL; i++)
	{
		birds[i] = pick_fittest();
	}

	console.log("generation "+ genCount);
	genCount++;
	current_gen = genCount;
	//empty last gen
	memBirds = [];
}

function pick_fittest(){
	
	var index = 0;
	var r = random(1);
	while(r > 0){
		// console.log(index);
		r = r-memBirds[index].fitness;
		index ++;
	}
	index--;
	let bird = memBirds[index];
	let child = new Bird(bird.brain);
	child.mutate();
	return child;
}

function calcFitness(){
	let sum=0;
	for (let bird of memBirds){
		sum+=bird.score;
	}
	for (let bird of memBirds){
		bird.fitness = bird.score / sum;
		if(bird.fitness > max_fitness)
			bestBird = bird;

	}

}