use rustymicrograd::{MultiLayerPerceptron, Value};

fn main() {
    let x = [2.0, 3.0, -1.0].map(|v| Value::new(v, None));
    let n = MultiLayerPerceptron::new(3, &[4, 4, 1]);
    let v = n.forward(&x);

    let preds = v.iter().map(|v| v.inner.borrow().data).collect::<Vec<_>>();
    println!("{:?}", preds);
}
