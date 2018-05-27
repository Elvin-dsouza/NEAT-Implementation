class Pipe{
	constructor(){
		this.spacing = 120;

		this.top = random(height/6,3/4*height);
		this.bottom = height - (this.top + this.spacing);
		this.x = width;
		this.w = 40;
		this.speed = 1.5;
	}

	show(){
		fill('#66b754');
		rect(this.x,0,this.w,this.top);
		rect(this.x,height-this.bottom,this.w,this.bottom);
	}

	update(){
		this.x -= this.speed;
	}

	offscreen()
	{
		if(this.x < - this.w)
		{
			return true;
		}
		else
			return false;
	}

	hits(bird){
		if(bird.x > this.x && bird.x < this.x + this.w)
			if(bird.y < this.top || bird.y > height - this.bottom)
			{
			return true;
			}
		else
			return false;
	}


}