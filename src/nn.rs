use rand::Rng;

use crate::Value;

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let w = (0..num_inputs)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0), None))
            .collect();
        let b = Value::new(rng.gen_range(-1.0..1.0), None);
        Neuron { w, b }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        x.iter()
            .zip(&self.w)
            .map(|(a, b)| a * b)
            .fold(self.b.clone(), |acc, v| &acc + &v)
            .tanh()
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut res = self.w.iter().collect::<Vec<_>>();
        res.push(&self.b);
        res
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Layer {
        let neurons = (0..num_outputs).map(|_| Neuron::new(num_inputs)).collect();
        Layer { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<&Value> {
        self.neurons
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }
}

pub struct MultiLayerPerceptron {
    pub layers: Vec<Layer>,
    pub sizes: Vec<usize>,
}

impl MultiLayerPerceptron {
    pub fn new(num_inputs: usize, layer_sizes: &[usize]) -> MultiLayerPerceptron {
        let mut sizes = vec![num_inputs];
        sizes.extend(layer_sizes);

        let inputs = &sizes[0..sizes.len() - 1];
        let outputs = &sizes[1..sizes.len()];

        let layers = inputs
            .iter()
            .zip(outputs)
            .map(|(num_inputs, num_outputs)| Layer::new(*num_inputs, *num_outputs))
            .collect();
        MultiLayerPerceptron { layers, sizes }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<&Value> {
        self.layers
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }
}
