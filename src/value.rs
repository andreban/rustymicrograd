use std::{cell::RefCell, fmt::Display, ops, rc::Rc};

pub type ValueInnerRef = Rc<RefCell<ValueInner>>;

#[derive(Clone, Debug)]
pub enum Op {
    None,
    Add(ValueInnerRef, ValueInnerRef),
    Mul(ValueInnerRef, ValueInnerRef),
    TanH(ValueInnerRef, f32),
    Pow(ValueInnerRef, f32),
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add(_, _) => write!(f, "+")?,
            Op::Mul(_, _) => write!(f, "*")?,
            Op::TanH(_, _) => write!(f, "tanh")?,
            Op::Pow(_, _) => write!(f, "pow")?,
            _ => {}
        };
        Ok(())
    }
}

#[derive(Debug)]
pub struct ValueInner {
    pub data: f32,
    pub op: Op,
    pub grad: f32,
    pub label: Option<&'static str>,
}

impl ValueInner {
    pub fn new(data: f32, op: Op, grad: f32, label: Option<&'static str>) -> ValueInnerRef {
        Rc::new(RefCell::new(ValueInner {
            data,
            op,
            grad,
            label,
        }))
    }

    pub fn backward(&self) {
        match &self.op {
            Op::Add(a, b) => {
                a.borrow_mut().grad += 1.0 * self.grad;
                b.borrow_mut().grad += 1.0 * self.grad;
                a.borrow().backward();
                b.borrow().backward();
            }
            Op::Mul(a, b) => {
                a.borrow_mut().grad += b.borrow().data * self.grad;
                b.borrow_mut().grad += a.borrow().data * self.grad;
                a.borrow().backward();
                b.borrow().backward();
            }
            Op::TanH(a, t) => {
                a.borrow_mut().grad += (1.0 - t.powf(2.0)) * self.grad;
                a.borrow().backward();
            }
            Op::Pow(a, b) => {
                let v = a.borrow().data.powf(b - 1.0);
                a.borrow_mut().grad += b * v * self.grad;
                a.borrow().backward();
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct Value {
    pub inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    pub fn new(data: f32, label: Option<&'static str>) -> Value {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                op: Op::None,
                grad: 0.0,
                label,
            })),
        }
    }

    pub fn tanh(&self) -> Value {
        let x = self.inner.borrow().data;
        let t = (f32::exp(2.0 * x) - 1.0) / (f32::exp(2.0 * x) + 1.0);
        let op = Op::TanH(self.inner.clone(), t);
        let grad = 0.0;
        Value {
            inner: ValueInner::new(x, op, grad, Some("tanh")),
        }
    }

    pub fn pow(&self, p: f32) -> Value {
        let data = self.inner.borrow().data;
        let out = data.powf(p);
        let op = Op::Pow(self.inner.clone(), p);
        let grad = 0.0;
        Value {
            inner: ValueInner::new(out, op, grad, Some("pow")),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Value(data = {})", self.inner.borrow().data)?;
        Ok(())
    }
}

impl ops::Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let data = self.inner.borrow().data + rhs.inner.borrow().data;
        let op = Op::Add(self.inner.clone(), rhs.inner.clone());
        let grad = 0.0;

        Value {
            inner: ValueInner::new(data, op, grad, Some("+")),
        }
    }
}

impl ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        ops::Add::<&Value>::add(&self, &rhs)
    }
}

impl ops::Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.inner.borrow().data * rhs.inner.borrow().data;
        let op = Op::Mul(self.inner.clone(), rhs.inner.clone());
        let grad = 0.0;

        Value {
            inner: ValueInner::new(data, op, grad, Some("*")),
        }
    }
}

impl ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        ops::Mul::<&Value>::mul(&self, &rhs)
    }
}

impl ops::Mul<f32> for Value {
    type Output = Value;
    fn mul(self, rhs: f32) -> Self::Output {
        let value = Value::new(rhs, None);
        self.mul(value)
    }
}

impl ops::Mul<f32> for &Value {
    type Output = Value;
    fn mul(self, rhs: f32) -> Self::Output {
        let value = Value::new(rhs, None);
        self.mul(&value)
    }
}

impl ops::Neg for &Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl ops::Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl ops::Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)
    }
}

impl ops::Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}
impl From<ValueInnerRef> for Value {
    fn from(inner: ValueInnerRef) -> Self {
        Value { inner }
    }
}

impl From<&ValueInnerRef> for Value {
    fn from(inner: &ValueInnerRef) -> Self {
        Value {
            inner: inner.clone(),
        }
    }
}
