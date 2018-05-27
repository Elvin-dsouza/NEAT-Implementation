
function preload() {
    
}
let car;
let car2;
function setup(){
	car = new Car(0);
	car2 = new Car(1);
	road = new Road();
	createCanvas(700,500);
	
}

function draw(){
	background(0);
	road.show();
	car2.show();
	car.show();

}


function keyPressed(){
	if(key == ' '){
	
	}
}

