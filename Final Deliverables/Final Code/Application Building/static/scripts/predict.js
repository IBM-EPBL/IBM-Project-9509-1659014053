var canvas   = document.getElementById('test'),
    ctx      = canvas.getContext('2d'),
    stack    = [],
    w        = window.innerWidth,
    h        = window.innerHeight;

var drawer = function(){
  //originally made background white
  //ctx.fillStyle="#222";
  ctx.clearRect(0,0,w,h);
  stack.forEach(function(el){
      el();  
  })
  requestAnimationFrame(drawer);
}
var anim = function(){
  var x = 0, y = 0;
  //tallness of blades * variable + min 
  var maxTall = Math.random()*(h/8)+(h/8);
  //thickness of bladess
  var maxSize = Math.random()*(h/60)+5;
  //speed of blade growth
  var speed = Math.random()*1;  
  var position = Math.random()*w-w/2;
  var c = function(l,u){return Math.round(Math.random()*(u||255)+l||0);}
  
  //color of grass   
  var color = 'rgb('+c(125,50)+','+c(225,80)+','+c(80,50)+')';
  return function(){
    
    //how fast + far the blades bend 
    var deviation=Math.cos(x/50)*Math.min(x/4,50),
        tall = Math.min(x/4,maxTall),
        size = Math.min(x/100,maxSize);
    x+=speed;
    ctx.save();
    
    ctx.strokeWidth=10;
    ctx.translate(w/2+position,h)
    ctx.fillStyle=color;
    
    ctx.beginPath();
    ctx.lineTo(-size,0);
    ctx.quadraticCurveTo(-size,-tall/2,deviation,-tall);
    ctx.quadraticCurveTo(size,-tall/2,size,0);
    //ctx.closePath();?
    ctx.fill();
    
    ctx.restore()
  }    
};
//number of blades it makes
for(var x = 0; x<(w/7);x++){stack.push(anim());}
canvas.width = w;
canvas.height = h;
drawer();



/*
Bouncing Balls orignally by Rob Glazebrook
Added glow, changed size, color and speed
*/

var particles = [],
    //number of particles
    particleCount = 200;
    Particle = function(x,y) {     
      this.x = x;
      this.y = y;
      
      
      
      //size of particles 
      this.radius = random(1,5);
      
      //colors red,green,blue,transparancy 
      this.rgba = 'rgba('+floor(random(240,245))+','+floor(random(219,245))+','+floor(random(140,144))+','+random(.2,.8)+')';
      
      //changes speed of particle
      this.vx = random(-.5,.5);
      this.vy = random(-.5,.5);
      
      // Draw our particle to the canvas.
      this.draw = function(ctx) {
        ctx.fillStyle = this.rgba;
        ctx.beginPath();
        ctx.arc(this.x,this.y,this.radius,0,TWO_PI);
        ctx.fill();
        
        //adds blinking glow
        ctx.shadowBlur = random(15,30);
        //glow color
        ctx.shadowColor = "white";
      };
      
      // Update our position. 
      this.update = function(ctx) {
      
        this.x += this.vx;
        this.y += this.vy;
        // Bounce off edges.
        if(this.x + this.radius > ctx.width) {
          this.vx *= -1;
          this.x = ctx.width - this.radius;
        }
        if(this.x - this.radius < 0) {
          this.vx *= -1;
          this.x = this.radius;
        }
        if(this.y + this.radius > ctx.height) {
          this.vy *= -1;
          this.y = ctx.height - this.radius;
        }
        if(this.y - this.radius < 0) {
          this.vy *= -1;
          this.y = this.radius;
        }        
      }
    };

var sketch = Sketch.create({
  setup: function() {
    var i = particleCount;
    while(i--) {
      var p = new Particle(random(0, this.width),random(0, this.height));
      particles.push(p);
    }
  },
  update: function() {
    var i = particleCount;
    while(i--) { 
      particles[i].update(this);
    }
  },
  draw: function() {
    var i = particleCount;
    while(i--) {
      particles[i].draw(this);
    }
  }
});

