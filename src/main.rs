use ndarray::{Array, Array2, Dimension};
use rand::{prelude::SliceRandom, Rng};

mod data;

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
#[derive(Debug)]
struct Network {
    layers: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    //Constructor
    fn new(layers: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            //Biases are initialized with random values
            biases: layers
                .iter()
                .skip(1)
                .map(|&s| {
                    (0..s)
                        .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-1.0..1.0))
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
                            .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-1.0..1.0))
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

    //Stochastic Gradient Descent function. Most basic algorithm for neural network training
    fn sgd(
        &mut self,
        training_data: &mut Vec<(Array2<f64>, Array2<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<&Vec<(Array2<f64>, usize)>>,
    ) {
        let mut rng = rand::thread_rng();

        //Main loop performing learning step with each epoch
        for j in 0..epochs {
            //Randomization of data for usage of mini-batch
            training_data.shuffle(&mut rng);

            //Generation of mini-batch vector. This is basicaly a collection of smaller datasets
            let mut mini_batches = training_data
                .windows(mini_batch_size)
                .step_by(mini_batch_size)
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<_>>>();

            //Sub-loob performing learning step for each of the mini-batch
            for mini_batch in &mut mini_batches {
                //dbg!(&mini_batch);
                self.update_mini_batch(mini_batch, eta)
             }

            //Data verification. Can be ommited
            if let Some(data) = &test_data {
                if j % 10 == 1 {
                    let output = self.evaluate(data);
                    println!("Epoch {}: {} / {}", j, output.1, output.0);
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
        let mut activation = x.clone();
        let mut activations = vec![x.clone()];
        let mut zs: Vec<Array2<f64>> = Vec::new();

        // Performing feedforward operation
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = sigmoid(z);
            activations.push(activation.clone());
        }

        // Calculating cost of the result
        let mut delta = Self::cost_derivative(activations.last().unwrap().clone(), y.clone())
            * sigmoid_prime(zs.last().unwrap().clone());


        // Setting up known values in nabla vectors
        *nabla_b.last_mut().unwrap() = delta.clone();
        //dbg!(delta);
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len() - 2].t());

        // Performing backward network pass
        for l in 2..self.layers.len() {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z.clone());

            delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;

            let b_len = nabla_b.len();
            let w_len = nabla_w.len();
            nabla_b[b_len - l] = delta.clone();
            nabla_w[w_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
        }

        // Returning calculated gradient vectors
        (nabla_b, nabla_w)
    }

    fn update_mini_batch(&mut self, mini_batch: &Vec<(Array2<f64>, Array2<f64>)>, eta: f64) {
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
            .map(|(w, nw)| w - nw * (eta / mini_batch.len() as f64) as f64)
            .collect();
        self.biases = self
            .biases
            .iter()
            .zip(nabla_b.iter())
            .map(|(b, nb)| b - nb * (eta / mini_batch.len() as f64) as f64)
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
    println!("Hello, world!");
    let mut x = Network::new(vec![16, 6, 6, 7]);
    //dbg!(&x);
    //dbg!(x.backprop(&Array2::<f64>::zeros((2, 1)), &Array2::<f64>::zeros((4, 1))));
    //Vector for learning data - added manually ATM
    let mut t_data: Vec<(Array2<f64>, Array2<f64>)>;
    //Vector for validation data - added manually ATM
    let mut v_data: Vec<Array2<f64>> = Vec::new();

    let data = std::fs::read_to_string("zoo.data");
    let animal_list = data::Animal::new_list(&data.as_ref().unwrap());
    let animal_list = animal_list
        .iter()
        .map(|a| a.into_training_arr2())
        .collect::<Vec<_>>();
    t_data = animal_list;

    //Adding learning data
    /*
    t_data.push((
        arr2(&[[0.0, 0.0]]).t().to_owned(),
        arr2(&[[0.0, 1.0]]).t().to_owned(),
    ));
    t_data.push((
        arr2(&[[0.0, 1.0]]).t().to_owned(),
        arr2(&[[1.0, 0.0]]).t().to_owned(),
    ));
    t_data.push((
        arr2(&[[1.0, 0.0]]).t().to_owned(),
        arr2(&[[1.0, 0.0]]).t().to_owned(),
    ));
    t_data.push((
        arr2(&[[1.0, 1.0]]).t().to_owned(),
        arr2(&[[0.0, 1.0]]).t().to_owned(),
    ));
    */

    v_data.push(t_data[0].0.clone());
    v_data.push(t_data[2].0.clone());
    v_data.push(t_data[6].0.clone());
    v_data.push(t_data[9].0.clone());
    v_data.push(t_data[15].0.clone());
    v_data.push(t_data[26].0.clone());
    v_data.push(t_data[12].0.clone());
    v_data.push(t_data[14].0.clone());

    //    dbg!(&t_data);
    let data_len = t_data.len();
    let test_data = t_data
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

    let animal_list = data::Animal::new_list(&data.unwrap());
    let animal_list = data::Animal::even_list(animal_list);
    let animal_list = animal_list
        .iter()
        .map(|a| a.into_training_arr2())
        .collect::<Vec<_>>();
    t_data = animal_list;

    println!("Learing record count: {}", t_data.len());
    x.sgd(&mut t_data, 10000, data_len / 2, 0.9, Some(&test_data));

    //Adding validation data
    /*
    v_data.push(arr2(&[[0.0, 0.0]]).t().to_owned());
    v_data.push(arr2(&[[0.0, 1.0]]).t().to_owned());
    v_data.push(arr2(&[[1.0, 0.0]]).t().to_owned());
    v_data.push(arr2(&[[1.0, 1.0]]).t().to_owned());
    v_data.push(arr2(&[[0.5, 0.7]]).t().to_owned());
    */

    for record in v_data.iter().enumerate() {
        let result = x.feed_forward(record.1.clone());
        /*println!(
            "For input data: X1: {}, X2: {}, network has outputted: Y1: {}, Y2: {}",
            &record[[0, 0]],
            &record[[1, 0]],
            &result[[0, 0]],
            &result[[1, 0]]
        );
        */
        println!("Result for test data {:?} : {:?}", v_data[record.0], result);
    }

    //let x =  Array::from_elem((1000,1000, 100), 1.);
    //dbg!(sigmoid(x));
}
