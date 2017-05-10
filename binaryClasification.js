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





// =============================================================================
// simple circuit
// =============================================================================

class Vars{
    constructor(){
        this.a = 1;
        this.b = -2; 
        this.c = -1;
    }
}

class Circuit{
    constructor(){
        this.mulg0 = new multiplyGate();
        this.mulg1 = new multiplyGate();
        this.addg0 = new addGate();
        this.addg1 = new addGate();
    }

    forward(x, y, a, b, c){
        this.ax = this.mulg0.forward(a, x); // a*x
        this.by = this.mulg1.forward(b, y); // b*y
        this.axpby = this.addg0.forward(this.ax, this.by); // a*x + b*y
        this.axpbypc = this.addg1.forward(this.axpby, c); // a*x + b*y + class
        return this.axpbypc;
    }

    backward(gradient_top){
        this.axpbypc.grad = gradient_top;
        this.addg1.backward(); // sets gradient in axpby and c
        this.addg0.backward(); // sets gradient in ax and by
        this.mulg1.backward(); // sets gradient in b and y
        this.mulg0.backward(); // sets gradient in a and x
    }
}







// =============================================================================
// Support Vectro Machine (SVM)
// =============================================================================

class SVM{
    constructor(){
        this.a = new Unit(1.0, 0.0);
        this.b = new Unit(-2.0, 0.0);
        this.c = new Unit(-1.0, 0.0);

        this.circuit = new Circuit();
    }

    forward(x, y){
        this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c);

        return this.unit_out;
    }

    backward(label){ // label is +1 or -1

        // reset pulls on a, b, c
        this.a.grad = 0.0;
        this.b.grad = 0.0;
        this.c.grad = 0.0;

        //compute the pull based on what the circuit output was
        var pull = 0.0;
        if(label === 1 && this.unit_out.value < 1){
            pull = 1; //the score was too low: pull up
        }
        if(label === -1 && this.unit_out.value > -1){
            pull = -1; //the score was too high for a positive example, pull down
        }
        this.circuit.backward(pull); //writes gradient into x, y, a, b, c

        //add regularization pull for parameters: towards zero and proportional to value
        this.a.grad += -this.a.value;
        this.b.grad += -this.b.value;
    }

    learnFrom(x, y, label){
        this.forward(x, y); // forward pass (set .value in all Units)
        this.backward(label); // backward pass (set .grad in all Units)
        this.parameterUpdate(); // parameters respond to tug
    }

    parameterUpdate(){
        var step_size = 0.01;
        this.a.value += step_size * this.a.grad;
        this.b.value += step_size * this.b.grad;
        this.c.value += step_size * this.c.grad;
    }
}






// =============================================================================
// begin training the SVM with Stochastic Gradient Descent
// =============================================================================

class TrainSVM{
    constructor(){
        this.data = [];
        this.labels = [];
        this.svm = new SVM();

        this.data.push([1.2, 0.7]);
        this.labels.push(1);

        this.data.push([-0.3, -0.5]);
        this.labels.push(-1);

        this.data.push([3.0, 0.1]);
        this.labels.push(1);

        this.data.push([-0.1, -1.0]);
        this.labels.push(-1);

        this.data.push([-1.0, 1.1]);
        this.labels.push(-1);

        this.data.push([2.1, -3]);
        this.labels.push(1);

        this.train();
    }

    train(){
        // the learning loop
        for(let iter = 0; iter < 400; iter++){
            // pick a random data point
            let i = Math.floor(Math.random() * this.data.length);
            let x = new Unit(this.data[i][0], 0.0);
            let y = new Unit(this.data[i][1], 0.0);
            let label = this.labels[i];
            this.svm.learnFrom(x, y, label);


            if(iter % 25 === 0){ // every 10 iterations
                console.log('training accuracy at iteration ' + iter + ": " + this.evaluateTrainingAccuracy())
            }
        }
    }

    /**
     * a funciton that computes the classification accuracy
     * 
     * 
     * @memberof TrainSVM
     */
    evaluateTrainingAccuracy(){
        var num_correct = 0;

        for(let i = 0; i < this.data.length; i++){
            let x = new Unit(this.data[i][0], 0.0);
            let y = new Unit(this.data[i][1], 0.0);
            let true_label = this.labels[i];

            // see if the prediction matches the provided label
            let predicted_label = this.svm.forward(x, y).value > 0 ? 1 : -1;
            if (predicted_label === true_label){
                num_correct++;
            } 
        }

        return num_correct / this.data.length;
    }
}

new TrainSVM();