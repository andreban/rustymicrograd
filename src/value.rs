use std::{cell::RefCell, fmt::Display, ops, rc::Rc};

pub type ValueInnerRef = Rc<RefCell<ValueInner>>;

/// Represents different operations that can be performed on a `Value`.
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

/// Represents the inner data of a `Value`.
#[derive(Debug)]
pub struct ValueInner {
    pub data: f32,
    pub op: Op,
    pub grad: f32,
    pub label: Option<&'static str>,
}

impl ValueInner {
    /// Creates a new `ValueInner` instance.
    ///
    /// # Arguments
    ///
    /// * `data` - The data value.
    /// * `op` - The operation associated with the value.
    /// * `grad` - The gradient value.
    /// * `label` - An optional label for the value.
    ///
    /// # Returns
    ///
    /// A `ValueInnerRef` reference to the newly created `ValueInner` instance.
    pub fn new(data: f32, op: Op, grad: f32, label: Option<&'static str>) -> ValueInnerRef {
        Rc::new(RefCell::new(ValueInner {
            data,
            op,
            grad,
            label,
        }))
    }

    /// Performs backward propagation of gradients for the value.
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

/// Represents a value in the computation graph.
///
/// The `Value` struct is used to store a scalar value along with its associated operations and gradients.
/// It is part of a simple automatic differentiation framework implemented in RustyMicroGrad.
#[derive(Clone, Debug)]
pub struct Value {
    pub inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    /// Creates a new `Value` instance.
    ///
    /// # Arguments
    ///
    /// * `data` - The data value.
    /// * `label` - An optional label for the value.
    ///
    /// # Returns
    ///
    /// A `Value` instance.
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

    /// Computes the hyperbolic tangent of the value.
    ///
    /// # Returns
    ///
    /// A new `Value` instance representing the hyperbolic tangent of the original value.
    pub fn tanh(&self) -> Value {
        let x = self.inner.borrow().data;
        let t = (f32::exp(2.0 * x) - 1.0) / (f32::exp(2.0 * x) + 1.0);
        let op = Op::TanH(self.inner.clone(), t);
        let grad = 0.0;
        Value {
            inner: ValueInner::new(t, op, grad, Some("tanh")),
        }
    }

    /// Computes the power of the value.
    ///
    /// # Arguments
    ///
    /// * `p` - The power value.
    ///
    /// # Returns
    ///
    /// A new `Value` instance representing the power of the original value.
    pub fn pow(&self, p: f32) -> Value {
        let data = self.inner.borrow().data;
        let out = data.powf(p);
        let op = Op::Pow(self.inner.clone(), p);
        let grad = 0.0;
        Value {
            inner: ValueInner::new(out, op, grad, Some("pow")),
        }
    }

    /// Performs backward propagation of gradients for the value.
    pub fn backward(&self) {
        self.inner.borrow().backward()
    }

    pub fn data(&self) -> f32 {
        self.inner.borrow().data
    }

    pub fn set_data(&self, data: f32) {
        self.inner.borrow_mut().data = data;
    }

    pub fn grad(&self) -> f32 {
        self.inner.borrow().grad
    }

    /// Sets the gradient for the value.
    pub fn set_grad(&self, grad: f32) {
        self.inner.borrow_mut().grad = grad;
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
        (&self).div(&rhs)
    }
}

impl ops::Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs.pow(-1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let a = Value::new(2.0, Some("a"));
        let b = Value::new(3.0, Some("b"));
        let c = a + b;
        assert_eq!(c.data(), 5.0);
        assert_eq!(c.grad(), 0.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Value::new(2.0, Some("a"));
        let b = Value::new(3.0, Some("b"));
        let c = a * b;
        assert_eq!(c.data(), 6.0);
        assert_eq!(c.grad(), 0.0);
    }

    #[test]
    fn test_tanh() {
        let a = Value::new(0.5, Some("a"));
        let b = a.tanh();
        assert!(b.data() - 0.46211717 < 0.0001);
        assert_eq!(b.grad(), 0.0);
    }

    #[test]
    fn test_pow() {
        let a = Value::new(2.0, Some("a"));
        let b = a.pow(3.0);
        assert_eq!(b.data(), 8.0);
        assert_eq!(b.grad(), 0.0);
    }

    #[test]
    fn test_backward() {
        let a = Value::new(2.0, Some("a"));
        let b = Value::new(3.0, Some("b"));
        let c = &a * &b;
        c.set_grad(1.0);
        c.backward();
        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
    }
}
