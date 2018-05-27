class Road{
	constructor(){
		this.x = 0;
		this.y = 200;
		this.rwidth = 100;

	}

	show(){
		fill(128);
		rect(this.x, this.y, width, this.rwidth);
	}
}