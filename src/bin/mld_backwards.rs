use rustymicrograd::{MultiLayerPerceptron, Value};

fn main() {
    let xs = [
        [2.0, 3.0, -1.0].map(|v| Value::new(v, None)),
        [3.0, -1.0, 0.5].map(|v| Value::new(v, None)),
        [0.5, 1.0, 1.0].map(|v| Value::new(v, None)),
        [1.0, 1.0, -1.0].map(|v| Value::new(v, None)),
    ];
    let ys = [1.0, -1.0, -1.0, 1.0].map(|v| Value::new(v, None)); // desired targets

    let nn = MultiLayerPerceptron::new(3, &[4, 4, 1]);

    // Run 1000 epochs.
    for _ in 1..100 {
        // Calculate predictions for the neural network.
        let predictions = xs
            .iter()
            .map(|x| nn.forward(x))
            .map(|mut v| v.pop().unwrap())
            .collect::<Vec<_>>();

        println!(
            "{:?}",
            predictions.iter().map(|p| p.data()).collect::<Vec<_>>()
        );

        // Calculate loss as the Squared Root Errors - the sum of pow((x - y), 2.0)
        let loss = ys
            .iter()
            .zip(predictions)
            .map(|(y, p)| (y - &p).pow(2.0))
            .fold(Value::new(0.0, None), |acc, v| acc + v);
        println!("{}", loss);

        // Reset gradients.
        nn.parameters().iter().for_each(|p| p.set_grad(0.0));

        // Calculate new gradients from loss.
        loss.set_grad(1.0);
        loss.backward();

        // Sets new values for the weights.
        nn.parameters().iter().for_each(|p| {
            let new_weight = p.data() - 0.05 * p.grad();
            p.set_data(new_weight);
        });
    }
}
