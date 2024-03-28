/// This module contains the implementation of a neural network.
mod nn;

/// This module contains the implementation of a value used in the neural network.
mod value;

pub use nn::*;
pub use value::*;

/// Prints the debug information of a given `Value`.
///
/// This function recursively traverses the `Value` and prints the debug information
/// of each operation it encounters. It also prints the label, data, and gradient of the `Value`.
///
pub fn debug(v: &Value) {
    let v = v.inner.borrow();
    match &v.op {
        Op::Add(v1, v2) => {
            debug(&v1.into());
            debug(&v2.into());
            println!("{}", v.op)
        }
        Op::Mul(v1, v2) => {
            debug(&v1.into());
            debug(&v2.into());
            println!("{}", v.op)
        }
        Op::TanH(v, _) => {
            debug(&v.into());
            println!("{}", v.borrow().op);
        }
        Op::Pow(v, _) => {
            debug(&v.into());
            println!("{}", v.borrow().op);
        }
        _ => {}
    }
    println!("{} | {} | {}", v.label.unwrap_or(""), v.data, v.grad);
}
