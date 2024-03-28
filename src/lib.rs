mod nn;
mod value;

pub use nn::*;
pub use value::*;

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
        _ => {}
    }
    println!("{} | {} | {}", v.label.unwrap_or(""), v.data, v.grad);
}
