const LEFT_LANE = 0;
const RIGHT_LANE = 1;
class Car{
	constructor(lane){
		if(lane === LEFT_LANE){
			this.x = 100;
			this.y = 210;
		}
		else if(lane === RIGHT_LANE){
			this.x = 100;
			this.y = 270;
		}
		
	}

	show(){
		fill("#ffddff");
		rect(this.x,this.y,50,20);
	}

	update(){

	}
		
}