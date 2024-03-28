use rand::Rng;

use crate::Value;

/// Represents a single neuron in a neural network.
pub struct Neuron {
    pub w: Vec<Value>, // Weights of the neuron
    pub b: Value,      // Bias of the neuron
}

impl Neuron {
    /// Creates a new neuron with random weights and bias.
    pub fn new(num_inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let w = (0..num_inputs)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0), None))
            .collect();
        let b = Value::new(rng.gen_range(-1.0..1.0), None);
        Neuron { w, b }
    }

    /// Performs the forward pass of the neuron.
    pub fn forward(&self, x: &[Value]) -> Value {
        x.iter()
            .zip(&self.w)
            .map(|(a, b)| a * b)
            .fold(self.b.clone(), |acc, v| &acc + &v)
            .tanh()
    }

    /// Returns the parameters (weights and bias) of the neuron.
    pub fn parameters(&self) -> Vec<&Value> {
        let mut res = self.w.iter().collect::<Vec<_>>();
        res.push(&self.b);
        res
    }
}

/// Represents a layer of neurons in a neural network.
pub struct Layer {
    pub neurons: Vec<Neuron>, // Neurons in the layer
}

impl Layer {
    /// Creates a new layer with the specified number of inputs and outputs.
    pub fn new(num_inputs: usize, num_outputs: usize) -> Layer {
        let neurons = (0..num_outputs).map(|_| Neuron::new(num_inputs)).collect();
        Layer { neurons }
    }

    /// Performs the forward pass of the layer.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    /// Returns the parameters (weights and biases) of the layer.
    pub fn parameters(&self) -> Vec<&Value> {
        self.neurons
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }
}

/// Represents a multi-layer perceptron neural network.
pub struct MultiLayerPerceptron {
    pub layers: Vec<Layer>, // Layers of the neural network
    pub sizes: Vec<usize>,  // Sizes of each layer
}

impl MultiLayerPerceptron {
    /// Creates a new multi-layer perceptron with the specified number of inputs and layer sizes.
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

    /// Performs the forward pass of the multi-layer perceptron.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Returns the parameters (weights and biases) of the multi-layer perceptron.
    pub fn parameters(&self) -> Vec<&Value> {
        self.layers
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }
}
