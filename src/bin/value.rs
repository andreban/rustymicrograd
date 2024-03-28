use rustymicrograd::{debug, Value};

fn main() {
    // inputs x1,x2
    let x1 = Value::new(2.0, Some("x1"));
    let x2 = Value::new(0.0, Some("x2"));

    // weights w1,w2
    let w1 = Value::new(-3.0, Some("w1"));

    let w2 = Value::new(1.0, Some("w2"));
    // bias of the neuron
    let b = Value::new(6.8813735870195432, Some("b"));

    // x1*w1 + x2*w2 + b
    let x1w1 = x1 * w1;
    x1w1.inner.as_ref().borrow_mut().label = Some("x1 * w1");

    let x2w2 = x2 * w2;
    x2w2.inner.as_ref().borrow_mut().label = Some("x2 * w2");

    let x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.inner.as_ref().borrow_mut().label = Some("x1 * w1 + x2 * w2");

    let n = x1w1x2w2 + b;
    n.inner.as_ref().borrow_mut().label = Some("n");

    let o = n.tanh();
    o.inner.as_ref().borrow_mut().label = Some("o");
    o.inner.as_ref().borrow_mut().grad = 1.0;
    o.inner.borrow().backward();

    debug(&o);
}
