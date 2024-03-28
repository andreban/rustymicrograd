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

    for _ in 1..100 {
        // Calculate predictions for the neural network.
        let predictions = xs
            .iter()
            .map(|x| nn.forward(x))
            .map(|mut v| v.pop().unwrap())
            .collect::<Vec<_>>();

        println!(
            "{:?}",
            predictions
                .iter()
                .map(|p| p.inner.as_ref().borrow().data)
                .collect::<Vec<_>>()
        );

        // Calculate loss as the Squared Root Errors - the sum of pow((x - y), 2.0)
        let loss = ys
            .iter()
            .zip(predictions)
            .map(|(y, p)| (y - &p).pow(2.0))
            .fold(Value::new(0.0, None), |acc, v| acc + v);
        println!("{}", loss);

        // Reset gradients.
        nn.parameters()
            .iter()
            .for_each(|p| p.inner.as_ref().borrow_mut().grad = 0.0);

        // Calculate new gradients from loss.
        loss.inner.as_ref().borrow_mut().grad = 1.0;
        loss.inner.as_ref().borrow().backward();

        nn.parameters().iter().for_each(|p| {
            let mut inner = p.inner.as_ref().borrow_mut();
            inner.data -= 0.01 * inner.grad;
        });
    }
}
