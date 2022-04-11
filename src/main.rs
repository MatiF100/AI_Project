use ndarray::{arr2, Array, Array2, Dimension};
use rand::{prelude::SliceRandom, Rng};

fn sigmoid<D>(z: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let mut z = z;
    z.iter_mut().for_each(|f| *f = 1.0 / (1.0 + (-*f).exp()));
    z
}

fn sigmoid_prime<D>(z: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let val = sigmoid(z);
    &val * (1.0 - &val)
}

#[derive(Debug)]
struct Network {
    layers: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(layers: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        Self {
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
            layers,
        }
    }

    fn feed_forward(&self, mut a: Array2<f64>) -> Array2<f64> {
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = sigmoid(w.dot(&a) + b);
        }
        return a;
    }

    fn sgd(
        &mut self,
        training_data: &mut Vec<(Array2<f64>, Array2<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<Vec<(Array2<f64>, usize)>>,
    ) {
        let mut rng = rand::thread_rng();

        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let mut mini_batches = training_data
                .windows(mini_batch_size)
                .step_by(mini_batch_size)
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<_>>>();
            for mini_batch in &mut mini_batches {
                //dbg!(&mini_batch);
                self.update_mini_batch(mini_batch, eta)
            }
            if let Some(data) = &test_data {
                let output = self.evaluate(data);
                println!("Epoch {}: {} / {}", j, output.1, output.0);
            } else {
                //println!("Epoch {} complete!", j);
            }
        }
    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
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

        let mut activation = x.clone();
        let mut activations = vec![x.clone()];
        let mut zs: Vec<Array2<f64>> = Vec::new();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = sigmoid(z);
            activations.push(activation.clone());
        }

        let mut delta = Self::cost_derivative(activations.last().unwrap().clone(), y.clone())
            * sigmoid_prime(zs.last().unwrap().clone());

        *nabla_b.last_mut().unwrap() = delta.clone();
        //dbg!(delta);
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len() - 2].t());

        for l in 2..self.layers.len() {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z.clone());

            delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;

            let b_len = nabla_b.len();
            let w_len = nabla_w.len();
            nabla_b[b_len - l] = delta.clone();
            nabla_w[w_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
        }

        (nabla_b, nabla_w)
    }

    fn update_mini_batch(&mut self, mini_batch: &Vec<(Array2<f64>, Array2<f64>)>, eta: f64) {
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

        for (x, y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            let x = nabla_w
                .iter()
                .zip(delta_nabla_w.iter())
                .map(|(nw, dnw)|  nw + dnw)
                .collect();
                nabla_w = x
        }

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

    fn cost_derivative(output_activations: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        output_activations - y
    }

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
    let mut x = Network::new(vec![2,2]);
    //dbg!(&x);
    //dbg!(x.backprop(&Array2::<f64>::zeros((2, 1)), &Array2::<f64>::zeros((4, 1))));
    //Vector for learning data - added manually ATM
    let mut t_data: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();
    //Vector for validation data - added manually ATM
    let mut v_data: Vec<Array2<f64>> = Vec::new();

    //Adding learning data
    t_data.push((arr2(&[[0.0, 0.0]]).t().to_owned(), arr2(&[[0.0, 1.0]]).t().to_owned()));
    t_data.push((arr2(&[[0.0, 1.0]]).t().to_owned(), arr2(&[[1.0, 0.0]]).t().to_owned()));
    t_data.push((arr2(&[[1.0, 0.0]]).t().to_owned(), arr2(&[[1.0, 0.0]]).t().to_owned()));
    t_data.push((arr2(&[[1.0, 1.0]]).t().to_owned(), arr2(&[[1.0, 0.0]]).t().to_owned()));

//    dbg!(&t_data);
    x.sgd(&mut t_data, 500, 4, 20.0, None);

    //Adding validation data
    v_data.push(arr2(&[[0.0, 0.0]]).t().to_owned());
    v_data.push(arr2(&[[0.0, 1.0]]).t().to_owned());
    v_data.push(arr2(&[[1.0, 0.0]]).t().to_owned());
    v_data.push(arr2(&[[1.0, 1.0]]).t().to_owned());
    v_data.push(arr2(&[[0.5, 0.7]]).t().to_owned());

    for record in &v_data{
        let result = x.feed_forward(record.clone());
        println!("For input data: X1: {}, X2: {}, network has outputted: Y1: {}, Y2: {}", &record[[0,0]], &record[[1,0]], &result[[0,0]], &result[[1,0]]);
    }

    //let x =  Array::from_elem((1000,1000, 100), 1.);
    //dbg!(sigmoid(x));
}
