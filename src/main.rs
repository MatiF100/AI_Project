use ndarray::{Array, Array2, Dimension};
use rand::{prelude::SliceRandom, Rng, SeedableRng};

use std::sync::mpsc;
use std::thread;

mod data;

// Main algorithm based on Neural Networks and Deep Learning by Michael Nielsen
// Math formulas and definitions slightly modified according to information provided by Prof. Jacek Kluska, and Prof. Roman Zajdel during lectures and individual consultation
// Extended with variable learning rate based on Neural Network Design by Hagan, M.T., H.B. Demuth, and M.H. Beale

const MAX_PERF_INC: f64 = 1.05;

//Sigmoidal function - basic activation function for neurons
fn sigmoid<D>(z: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let mut z = z;
    z.iter_mut().for_each(|f| *f = 1.0 / (1.0 + (-*f).exp()));
    z
}

//Derivative of sigmoidal function
fn sigmoid_prime<D>(z: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let val = sigmoid(z);
    &val * (1.0 - &val)
}

//Implementation of neural network. Includes both, learning algorithms and the usage of network itself
#[derive(Debug, Clone)]
struct Network {
    layers: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
    name: String,
}

impl Network {
    //Constructor
    fn new(layers: Vec<usize>) -> Self {
        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(167788);
        let layers = layers
            .into_iter()
            .filter(|x| *x > 0)
            .collect::<Vec<usize>>();
        Self {
            name: String::from("Network 0"),
            //Biases are initialized with random values
            biases: layers
                .iter()
                .skip(1)
                .map(|&s| {
                    (0..s)
                        .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-0.5..0.5))
                        .collect::<Vec<f64>>()
                })
                .map(|v| Array2::from_shape_vec((v.len(), 1), v).unwrap())
                .collect(),

            //Weights are also initialized with random values
            weights: layers
                .windows(2)
                .map(|x| {
                    (
                        (x[0], x[1]),
                        (0..x[0] * x[1])
                            .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-0.5..0.5))
                            .collect::<Vec<f64>>(),
                    )
                })
                .map(|(x, v)| Array2::from_shape_vec((x.1, x.0), v).unwrap())
                .collect::<Vec<_>>(),
            //Layers are moved in from the argument
            layers,
        }
    }

    //Function with vector multiplication for all of the layers.
    //It's arguments are only inputs for the first, "dummy" layer of neurons
    fn feed_forward(&self, mut a: Array2<f64>) -> Array2<f64> {
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = sigmoid(w.dot(&a) + b);
        }
        return a;
    }

    // Total quadratic cost of a network for given training data
    // For single output neuron, or entire neuron layer if we use matrixes, the formula is (expected_out - output)^2 / 2
    fn mse(&self, training_data: &Vec<(Array2<f64>, Array2<f64>)>) -> f64 {
        training_data
            .iter()
            // Performing forward pass to determine network's outputs for entire training set
            .map(|v| (self.feed_forward(v.0.clone()), v.1.clone()))
            // Calculating error for each training pair (quadratic cost)
            .map(|v| (v.0 - v.1).fold(0.0, |_, val| val.powf(2.0)) / 2.0)
            // Calculating Mean Squared Error - average squared cost over all training pairs
            .map(|e| e.powf(2.0))
            .sum::<f64>()
            / training_data.len() as f64
    }

    //Stochastic Gradient Descent function. Most basic algorithm for neural network training
    fn sgd(
        &mut self,
        training_data: &mut Vec<(Array2<f64>, Array2<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        mut eta: f64,
        test_data: Option<&Vec<(Array2<f64>, usize)>>,
        eta_mod: Option<(f64, f64)>,
        target_cost: f64,
        report_interval: usize,
    ) {
        let mut rng = rand::thread_rng();

        //Main loop performing learning step with each epoch
        for j in 1..=epochs {
            //Randomization of data for usage of mini-batch
            training_data.shuffle(&mut rng);

            //Generation of mini-batch vector. This is basicaly a collection of smaller datasets
            let mut mini_batches = training_data
                .windows(mini_batch_size)
                .step_by(mini_batch_size)
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<_>>>();

            let batch_count = mini_batches.len();
            // Branching based on existance of adaptive learning rate parameters
            match eta_mod {
                Some((dec, inc)) => {
                    // Saving state of network before readjustment of weights and biases
                    let saved_weights = self.weights.clone();
                    let saved_biases = self.biases.clone();
                    let previous_error = self.mse(training_data);

                    //Sub-loob performing learning step for each of the mini-batch
                    for mini_batch in &mut mini_batches {
                        //dbg!(&mini_batch);
                        self.update_mini_batch(mini_batch, eta, batch_count)
                    }

                    // Verification of newly achieved Mean Square Error
                    let new_error = self.mse(training_data);
                    if new_error < target_cost {
                        if let Some(data) = &test_data {
                            let output = self.evaluate(data);
                            println!("{},{}", self.name, output.1 as f64 / output.0 as f64);
                        }
                        break;
                    }
                    if new_error > previous_error * MAX_PERF_INC {
                        // Restoring backup
                        //self.weights = saved_weights;
                        //self.biases = saved_biases;

                        // Adaptation - learning rate decreases
                        eta *= dec;
                    } else if new_error < previous_error {
                        // Adaptation - learning rate increases
                        eta *= inc;
                    }
                    // else statement does nothing - ommited
                }
                None => {
                    //Sub-loob performing learning step for each of the mini-batch
                    for mini_batch in &mut mini_batches {
                        //dbg!(&mini_batch);
                        self.update_mini_batch(mini_batch, eta, batch_count);
                        let new_error = self.mse(training_data);
                        if new_error < target_cost {
                            if let Some(data) = &test_data {
                                let output = self.evaluate(data);
                                println!("{},{}", self.name, output.1 as f64 / output.0 as f64);
                            }
                            break;
                        }
                    }
                }
            }

            //Data verification. Can be ommited
            if let Some(data) = &test_data {
                if j % report_interval == 0 && report_interval != 0 {
                    let output = self.evaluate(data);
                    println!("{},{}", self.name, output.1 as f64 / output.0 as f64);
                }
            } else {
                //println!("Epoch {} complete!", j);
            }
        }
    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        //Initialization of gradient vectors.
        let mut nabla_b = self
            .biases
            .iter()
            .map(|b| Array::zeros(b.raw_dim()))
            .collect::<Vec<Array2<f64>>>();
        let mut nabla_w = self
            .weights
            .iter()
            .map(|w| Array::zeros(w.raw_dim()))
            .collect::<Vec<Array2<f64>>>();

        // Preparing initial information for forward network pass
        // Because of lifetimes and loop scope further in the function, it is best to make copies of input matrix here
        let mut activation = x.clone();
        let mut activations = vec![activation];

        // zs is a Vector of neurons non linear blocks inputs - these will be calculated in the following loop
        let mut zs: Vec<Array2<f64>> = Vec::new();

        // Performing feedforward operation
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            // z is the input of non-linear block, necessary in the following gradient calculation
            let z = w.dot(activations.iter().last().unwrap()) + b;
            zs.push(z.clone());

            // As the current matrix of non-linear block inputs is calculated, it is passed as argument to the activation function
            // TODO: in this place other activation functions can be implemented
            activation = sigmoid(z);

            // Saving the outputs of neuron layer, for later use
            activations.push(activation);
        }

        // Calculating per_class or per_output cost of the result at this point, it is also worth noting that the "delta" is only partially calculated
        // With used notation, the delta itself does not include the eta, or learning rate
        // Cost derivative function only calculates error, or difference between achieved and expected output
        // TODO: possibly unnecessary function call
        let mut delta = Self::cost_derivative(activations.last().unwrap().clone(), y.clone())
            * sigmoid_prime(zs.last().unwrap().clone());

        // Setting up known values in gradient vectors
        // Last layer is easiest to calculate, as it does not require any data not available at the moment
        // We they will be used as we perform the backward pass, calculating bias and weight gradients for every layer
        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len() - 2].t());

        // Performing backward network pass
        // Side note: if the book gives example of any identifier as "l" "I" or "L" one should never follow the book and come up with anything that differs from 1
        for idx in 2..self.layers.len() {
            // Getting the input of non-linear block for the idx-th layer counting from the end
            let z = &zs[zs.len() - idx];

            // Calculating the derivative of activation function for given input
            let derivative = sigmoid_prime(z.clone());

            // Calculating delta - gradient for given layer
            // TODO: Include the generic formula into readme
            delta = self.weights[self.weights.len() - idx + 1].t().dot(&delta) * derivative;

            // Boilerplate forced by borrow-checker. Since .len() uses immutable reference, it would block the assignment operation if used inline
            // Works fine this way though, since usize implements "Copy"
            let b_len = nabla_b.len();
            let w_len = nabla_w.len();

            // Actual gradient for biases and weights is pretty similar
            // The difference is that weight gradient is additionally multiplied by the activation state of given layer
            nabla_b[b_len - idx] = delta.clone();
            nabla_w[w_len - idx] = delta.dot(&activations[activations.len() - idx - 1].t());
        }

        // Returning calculated gradient vectors
        (nabla_b, nabla_w)
    }

    fn update_mini_batch(
        &mut self,
        mini_batch: &Vec<(Array2<f64>, Array2<f64>)>,
        eta: f64,
        batches_count: usize,
    ) {
        // Allocation of gradient vectors
        let mut nabla_b = self
            .biases
            .iter()
            .map(|b| Array::zeros(b.raw_dim()))
            .collect::<Vec<Array2<f64>>>();
        let mut nabla_w = self
            .weights
            .iter()
            .map(|w| Array::zeros(w.raw_dim()))
            .collect::<Vec<Array2<f64>>>();

        // Loop performing learning iteration over all mini_batches
        for (x, y) in mini_batch {
            // Getting updated gradients from backpropagation algorithm
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);

            // Calculating new gradients with respect to ones created in first steps and also newly calculated ones
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = nabla_w
                .iter()
                .zip(delta_nabla_w.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }

        // Calculating new values for weights and biases based on recieved gradients with respect to batch size and learning rate
        self.weights = self
            .weights
            .iter()
            .zip(nabla_w.iter())
            .map(|(w, nw)| w - nw * (eta / batches_count as f64) as f64)
            .collect();
        self.biases = self
            .biases
            .iter()
            .zip(nabla_b.iter())
            .map(|(b, nb)| b - nb * (eta / batches_count as f64) as f64)
            .collect();
    }

    // Helper function used to calculate "cost" or error created by neural network at a given moment
    fn cost_derivative(output_activations: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        output_activations - y
    }

    // Helper function used to compare network's output with expected values
    // Can also be called "verification function"
    fn evaluate(&self, test_data: &Vec<(Array2<f64>, usize)>) -> (usize, usize) {
        let mut local_data = test_data.clone();
        let x = local_data
            .iter_mut()
            .map(|(x, y)| {
                (
                    self.feed_forward(x.clone())
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(index, _)| index),
                    y,
                )
            })
            .filter(|(a, b)| a.unwrap_or(0) == **b)
            .count();
        (test_data.len(), x)
    }
}

fn main() {
    let mut x = Network::new(vec![2,2]);
    let or = vec![
        (
            Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap(),
        ),
        (
            Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap(),
        ),
        (
            Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap(),
        ),
        (
            Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap(),
        ),
    ];
    let test_data = or
        .iter()
        .map(|(input, output)| {
            (
                input,
                output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap(),
            )
        })
        .map(|(a, s)| (a.clone(), s.0))
        .collect::<Vec<_>>()
        .to_owned();
    x.sgd(
        &mut or.clone(),
        2000,
        4,
        0.6,
        Some(&test_data),
        //None,
        Some((0.95, 1.02)),
        0.25 / 8.0,
        1,
    );
    dbg!(x.feed_forward(Array2::<f64>::from_shape_vec((2, 1), vec![-5.0,0.0]).unwrap()));
    dbg!(x.feed_forward(Array2::<f64>::from_shape_vec((2, 1), vec![0.0,1.3]).unwrap()));
    dbg!(x.feed_forward(Array2::<f64>::from_shape_vec((2, 1), vec![1.0,0.0]).unwrap()));
    dbg!(x.feed_forward(Array2::<f64>::from_shape_vec((2, 1), vec![1.0,1.0]).unwrap()));
    //let x = Network::new(vec![16, 8, 7]);

    //dbg!(&x);
    //dbg!(x.backprop(&Array2::<f64>::zeros((2, 1)), &Array2::<f64>::zeros((4, 1))));

    //Vector for learning data
    let t_data: Vec<(Array2<f64>, Array2<f64>)>;
    //Vector for validation data
    let v_data: Vec<(Array2<f64>, Array2<f64>)>;

    let data = std::fs::read_to_string("zoo.data");

    let animal_list = data::Animal::new_list(&data.unwrap());
    let animal_list = data::Animal::partitioned_list(animal_list, 0.75);
    t_data = animal_list
        .0
        .iter()
        .map(|a| a.into_training_arr2())
        .collect::<Vec<_>>();

    v_data = animal_list
        .1
        .iter()
        .map(|a| a.into_training_arr2())
        .collect::<Vec<_>>();

    //    dbg!(&t_data);
    let data_len = t_data.len();
    let test_data = v_data
        .iter()
        .map(|(input, output)| {
            (
                input,
                output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap(),
            )
        })
        .map(|(a, s)| (a.clone(), s.0))
        .collect::<Vec<_>>()
        .to_owned();

    println!("Learning record count: {}", t_data.len());
    let lr_step = vec![
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 10.0,
    ];
    let lr_inc_step = vec![
        1.07, 1.08, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4,
    ];
    let lr_dec_step = vec![
        0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
    ];

    let threads = std::sync::Arc::new(std::sync::Mutex::new(0));
    let (sync_tx, sync_rx) = mpsc::sync_channel(12);
    let (tx, rx) = mpsc::channel();

    let tmp_threads = threads.clone();
    thread::spawn(move || {
        for s1 in 8..t_data.len() / 3 - 7 {
            for s2 in 8..(t_data.len() / 3 - 7 - s1) {
                if s2 == 0 && s1 != 0{
                    continue;
                }
                let x = Network::new(vec![16, s1, s2, 7]);
                for lr in &lr_step {
                    for lr_dec in &lr_dec_step {
                        for lr_inc in &lr_inc_step {
                            let mut net = x.clone();
                            net.name = format!("{},{},{},{},{}", s1, s2, lr, lr_dec, lr_inc);
                            let mut t_data = t_data.clone();
                            let test_data = test_data.clone();

                            let local_lrs = (*lr, *lr_dec, *lr_inc);
                            let local_sync_tx = sync_tx.clone();
                            let local_tx = tx.clone();

                            local_sync_tx.send(()).unwrap();
                            let inner_threads = std::sync::Arc::clone(&tmp_threads);

                            *inner_threads.lock().unwrap() += 1;
                            thread::spawn(move || {
                                net.sgd(
                                    &mut t_data,
                                    10000,
                                    data_len,
                                    local_lrs.0,
                                    Some(&test_data),
                                    //None
                                    Some((local_lrs.1, local_lrs.2)),
                                    0.25 / data_len as f64,
                                    //0.00001,
                                    10000,
                                );
                                local_tx.send(()).unwrap();
                            });
                        }
                    }
                }
            }
        }
    });

    loop {
        let threads = std::sync::Arc::clone(&threads);
        //println!("{}", threads.lock().unwrap());
        rx.recv().unwrap();
        //println!("{}", threads.lock().unwrap());
        sync_rx.recv().unwrap();
        *threads.lock().unwrap() -= 1;
        if *threads.lock().unwrap() <= 0 {
            break;
        }
    }
    /*
    x.sgd(
        &mut t_data,
        50000,
        data_len / 2,
        0.99,
        Some(&test_data),
        None, //Some((0.7, 1.15)),
    );
    */

    /*
    let mut y = x.clone();
    let mut t_data_2 = t_data.clone();
    let test_data_2 = test_data.clone();

    thread::spawn(move ||
        x.sgd(
        &mut t_data,
        50000,
        data_len / 2,
        0.9,
        Some(&test_data),
        //None
        Some((0.9, 1.05)),
    )

    );

    thread::spawn(move ||
        y.sgd(
        &mut t_data_2,
        50000,
        data_len / 2,
        0.9,
        Some(&test_data_2),
        None
        //Some((0.9, 1.05)),
    )
    );

    loop{}
    */
    /*
    for record in v_data.iter().enumerate() {
        let result = x.feed_forward(record.1.clone());

        println!("Result for test data {:?} : {:?}", v_data[record.0], result);
    }

    */
    //let x =  Array::from_elem((1000,1000, 100), 1.);
    //dbg!(sigmoid(x));
}
