var X = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] // array of 2-dimensional data
var y = [1, -1, 1] // array of labels
var w = [0.1, 0.2, 0.3] // example: random numbers
var alpha = 0.1; // regularization strength


class Cost{
    constructor(X, y, w){
        this.total_cost = 0.0;
        this.N = X.length;
    }

    loop(){
        for(let i = 0; i < this.N; i++){
            //loop over all data points and compute their score
            let xi = X[i];
            let score = w[0] * xi[0] + w[1] * xi[1] + w[2];

            //accumulate cost based on how compatible the score is with the label
            let yi = y[i];
        }
    }
}