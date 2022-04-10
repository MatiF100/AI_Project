use ndarray::{Array, Array1, Array2, Dimension};
use rand::{Rng, prelude::SliceRandom};


fn sigmoid<D>(z: Array<f64, D>) -> Array<f64, D>
where D: Dimension{
    let mut z = z;
    z.iter_mut().for_each(|f|  *f = 1.0/(1.0 + (-*f).exp()));
    z
}
#[derive(Debug)]
struct Network {
    layers: Vec<usize>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(layers: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            biases: layers
                .iter().skip(1)
                .map(|&s| {
                    (0..s)
                        .map(|_| rng.gen_range::<f64, std::ops::Range<f64>>(-5.0..5.0))
                        .collect::<Vec<f64>>()
                })
                .map(|v| Array1::from(v))
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
                .map(|(x, v)| Array2::from_shape_vec((x.0, x.1), v).unwrap())
                .collect::<Vec<_>>(),
            layers,
        }
    }

    fn feed_forward(&self, a: &mut Array2<f64>){
        for (b,w) in self.biases.iter().zip(self.weights.iter()){
            *a = sigmoid(w.dot(a) + b);
        }
    }

    fn sgd(&self, training_data: &mut Vec<(f64,f64)>, epochs: usize, mini_batch_size: usize, eta: usize, test_data: Option<&Vec<(f64,f64)>>){
        let mut rng = rand::thread_rng();
        if let Some(_data) = test_data{
            //Testing here
        }

        for j in 0..epochs{
            training_data.shuffle(&mut rng);
            let mut mini_batches = training_data.windows(mini_batch_size).step_by(mini_batch_size).map(|s| s.to_vec()).collect::<Vec<Vec<_>>>();
            for mut mini_batch in &mut mini_batches{
                //self.update_mini_batch(&mut mini_batch, eta)
            }
        }
    }

    fn backprop(&self, x: Array1<f64>, y: Array1<f64>){
        let nabla_b = self.biases.iter().map(|b| Array::zeros(b.raw_dim())).collect::<Vec<Array1<f64>>>();
        let nabla_w = self.weights.iter().map(|w| Array::zeros(w.raw_dim())).collect::<Vec<Array2<f64>>>();

        let mut activation = x.clone();
        let mut activations = vec![x];
        let mut zs: Vec<Array1<f64>>= Vec::new();

        for (b,w) in self.biases.iter().zip(self.weights.iter()){
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = sigmoid(z);
            activations.push(activation.clone());
        }

    }
}

fn main() {
    println!("Hello, world!");
    let x = Network::new(vec![2, 5, 6, 4]);
    dbg!(x);

    //let x =  Array::from_elem((1000,1000, 100), 1.);
    //dbg!(sigmoid(x));
}
