use ndarray::{Array, Array1, Array2, Dim, Dimension};
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
                        .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-5.0..5.0))
                        .collect::<Vec<f64>>()
                })
                .map(|v| Array2::from_shape_vec((v.len(),1),v).unwrap())
                .collect(),
            weights: layers
                .windows(2)
                .map(|x| {
                    (
                        (x[0], x[1]),
                        (0..x[0] * x[1])
                            .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-5.0..5.0))
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
        training_data: &mut Vec<(Array1<f64>, Array1<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: usize,
        test_data: Option<&Vec<(Array1<f64>, Array1<f64>)>>,
    ) {
        let mut rng = rand::thread_rng();
        if let Some(_data) = test_data {
            //Testing here
        }
        let n = training_data.len();

        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let mut mini_batches = training_data
                .windows(mini_batch_size)
                .step_by(mini_batch_size)
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<_>>>();
            for mut mini_batch in &mut mini_batches {
                //self.update_mini_batch(&mut mini_batch, eta)
            }
        }
    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>){
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
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len()-3].t());
    
        for l in 2..self.layers.len(){
            let z = &zs[zs.len()-l];
            let sp = sigmoid_prime(z.clone());

            delta = self.weights[self.weights.len() -l + 1].t().dot(&delta) * sp;

            let b_len = nabla_b.len();
            let w_len = nabla_w.len();
            nabla_b[b_len - l] = delta.clone();
            nabla_w[w_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
        }

        (nabla_b,nabla_w)

    }
    
    fn update_mini_batch(&mut self, mini_batch: &Vec<(Array2<f64>, Array2<f64>)>, eta: usize){
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

        for (x,y) in mini_batch{
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            nabla_b = nabla_b.iter().zip(delta_nabla_b.iter()).map(
                |(nb, dnb)| nb+dnb
            ).collect();
            nabla_w = nabla_w.iter().zip(delta_nabla_w.iter()).map(
                |(nw, dnw)| nw+dnw
            ).collect();
        }

        self.weights = self.weights.iter().zip(nabla_w.iter()).map(|(w,nw)| w - nw*(eta/mini_batch.len()) as f64).collect();
        self.biases = self.biases.iter().zip(nabla_b.iter()).map(|(b,nb)| b - nb*(eta/mini_batch.len()) as f64).collect();

    }

    fn cost_derivative(output_activations: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        output_activations - y
    }

    /*
    fn evaluate(&self, test_data: &Vec<(f64,f64)>) -> f64{
        test_data.iter().map(
            |(x, y)| (self.feed_forward(x).iter().max().unwrap(), y)
        );
        0.1
    }
    */
}

fn main() {
    println!("Hello, world!");
    let x = Network::new(vec![2, 5, 6, 4]);
    dbg!(&x);
    dbg!(x.backprop(&Array2::<f64>::zeros((2, 1)), &Array2::<f64>::zeros((4, 1))));

    //let x =  Array::from_elem((1000,1000, 100), 1.);
    //dbg!(sigmoid(x));

}
