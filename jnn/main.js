const TOTAL = 350;
let cycles = 100;
let birds = [];
let pipes = [];
let memBirds = [];
let counter = 0;
let slider;
let current_gen = 0;
let max_score = 0;
let img;
function preload() {
    img = loadImage('flappy.svg');
}

function setup(){
	createCanvas(400,600);
	slider = createSlider(1,100,1);
	for(let i = 0; i < TOTAL; i++)
 		birds[i]= new Bird();
 	// pipes.push(new Pipe());
	
}

function draw(){
	// background(0);

	for(let n = 0; n< slider.value(); n++)
	{
		if(counter % 210 == 0)
		{
			pipes.push(new Pipe());

		}

	
		counter ++;
		for(let i =pipes.length-1; i >= 0; i--){
			// pipes[i].show();
		    pipes[i].update();
		    for( let j = birds.length-1; j >=0 ; j--)
		    {
		    	 if(pipes[i].hits(birds[j])){
		    	 	memBirds.push(birds.splice(j,1)[0]);
			    	// console.log("BIRDSTRIKE");

			    }

		    }

		    if(pipes[i].offscreen()){
		    	pipes.splice(i,1);	    
		    } 


	 }
	  for( let j = birds.length-1; j >=0 ; j--)
	  {
		     if(birds[j].hitsBottom()){
			    	memBirds.push(birds.splice(j,1)[0]);
			    }
	  }
		   
	


	for(bird of birds)
	{
		// bird.show();
		bird.think(pipes);
		bird.update();
	}
	if(birds.length === 0)
	{
	 	counter = 0;
	 	pipes = [];
	 	nextGeneration();
	}
	}	
	
	
	background('#5d9afc');
	textSize(24);
	fill('#ffffff');
	textStyle(BOLD);
	text('Gen: '+current_gen, 10, 30);
	
	
	for(let bird of birds){
		bird.show();
	}

	for(let pipe of pipes)
	{
		pipe.show();
	}
	if(bestBird)
	drawnn(bestBird);
	// for()
	
	// ellipse(25, height/2,20,20);
}


function keyPressed(){
	if(key == ' '){
		bird.up();
		console.log("space");
	}
}

