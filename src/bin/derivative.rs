fn main() {
    let h = 0.00001;
    let a = 2.0;
    let b = -3.0;
    let c = 10.0;
    let d1 = a * b + c;

    let a = a + h;
    let d2 = a * b + c;
    let slope = (d2 - d1) / h;

    println!("d1: {d1}, d2: {d2}, slope: {slope}");
}
