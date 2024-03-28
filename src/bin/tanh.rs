use rustymicrograd::Value;

fn main() {
    let x = Value::new(0.5, Some("x"));
    let xtanh = x.tanh();
    println!("{}", xtanh);
}
