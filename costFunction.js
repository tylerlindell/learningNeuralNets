var X = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] // array of 2-dimensional data
var y = [1, -1, 1] // array of labels
var w = [0.1, 0.2, 0.3] // example: random numbers
var alpha = 0.1; // regularization strength


/**
 * cost function quantifies the SVM's unhappiness
 * 
 * A cost function is an expression that measuress how bad your classifier is. 
 * When the training set is perfectly classified, the cost (ignoring the regularization) will be zero.
 * http://karpathy.github.io/neuralnets/
 * 
 * @class Cost
 */
class Cost{
    constructor(X, y, w, alpha){
        this.total_cost = 0.0;
        this.N = X.length;
        this.reg_cost;
        let _self = this;
        this.loop(() => this.regularization())
    }

    loop(callback){
        for(let i = 0; i < this.N; i++){
            //loop over all data points and compute their score
            let xi = X[i];
            let score = w[0] * xi[0] + w[1] * xi[1] + w[2];

            //accumulate cost based on how compatible the score is with the label
            let yi = y[i]; //label
            let costi = Math.max(0, -yi * score + 1);
            console.log('example ' + i + ': xi = (' + xi + ' and label = ' + yi);
            console.log('  score computed to be ' + score.toFixed(3));
            console.log('  => cost computed to be ' + costi.toFixed(3));
            this.total_cost += costi;
        }

        callback();
    }

    /**
     * regularization cost: we want small weights
     * 
     * 
     * @memberof Cost
     */
    regularization(){
        this.reg_cost = alpha * (w[0]*w[0] + w[1]*w[1])
        console.log('total cost is ' + this.total_cost.toFixed(3))
        return this.total_cost;
    }
}

new Cost(X, y, w, alpha);