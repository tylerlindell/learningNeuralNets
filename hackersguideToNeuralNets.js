// every Unit corresponds to a wire in the diagrams
var Unit = function(value, grad) {
    // value computed in the forward pass
    this.value = value;
    // the derivative of circuit output w.r.t this unit, computed in backward pass
    this.grad = grad;
}




// =============================================================================
// gates
// =============================================================================


/**
 * multiply two values
 */
var multiplyGate = function()  {};
multiplyGate.prototype = {
    // multiply values and store in Unit for wire
    forward: function(u0, u1) {
        // store pointers to input Units u0 and u1 and output unit utop
        this.u0   = u0;
        this.u1   = u1;
        this.utop = new Unit(u0.value * u1.value, 0.0);
        return this.utop;
    },
    // populate the gradient of Unit
    backward: function()  {
        // take the gradient in output unit and chain it with the
        // local gradients, which we derived for multiply gate before
        // then write those gradients to those Units.
        this.u0.grad += this.u1.value * this.utop.grad;
        this.u1.grad += this.u0.value * this.utop.grad;
    }
}



/**
 * add two values
 */
var addGate = function(){};
addGate.prototype = {
    // add values and store in Unit for wire
    forward: function(u0, u1) {
        this.u0   = u0;
        this.u1   = u1;
        this.utop = new Unit(u0.value + u1.value, 0.0);
        return this.utop;
    },
    // populate the gradient of Unit
    backward: function() {
        // add gate. derivative wrt both inputs is 1
        this.u0.grad += 1 * this.utop.grad;
        this.u1.grad += 1 * this.utop.grad;
    }
}



/**
 * Sigmoid function is defined as:
 */
var sigmoidGate = function() {
    // helper function (squishing function - keeps numbers between zero and one)
    this.sig = function(x) { 
        return 1 / (1 + Math.exp(-x));
    }
}
sigmoidGate.prototype ={
    //squish value to be greater than zero and less than one then save value to Unit for wire 
    forward: function(u0) {
        this.u0   = u0;
        this.utop = new Unit(this.sig(this.u0.value), 0.0);
        return this.utop;
    },
    // populate the gradient of Unit
    backward: function() {
        var s = this.sig(this.u0.value);
        //The gradient with respect to its single input, as you can check on Wikipedia or derive yourself if you know some calculus is given by this expression: (s * (1 - s))
        this.u0.grad += (s * (1 - s)) * this.utop.grad;
    }
}


// =============================================================================
// lets try it out
// =============================================================================

// create input units
let a = new Unit(1.0, 0.0),
    b = new Unit(2.0, 0.0),
    c = new Unit(-3.0, 0.0),
    x = new Unit(-1.0, 0.0),
    y = new Unit(3.0, 0.0);

let ax, by, axpby, axpbypc, s;

// create gates
let mulg0 = new multiplyGate(),
    mulg1 = new multiplyGate(),
    addg0 = new addGate(),
    addg1 = new addGate(),
    sg0   = new sigmoidGate();

    
// do the forward pass
var forwardNeuron = function() {
    ax      = mulg0.forward(a, x);      // a * x = -1
    by      = mulg1.forward(b, y);      // b * y = 6
    axpby   = addg0.forward(ax, by);    // a * x + b * y = 5
    axpbypc = addg1.forward(axpby, c);  // a * x + b * y + c = 2
    s       = sg0.forward(axpbypc);     // sig(a * x + b * y + c) = 0.8808
}

forwardNeuron();

console.log('circuit output:', s.value); // prints 0.8808