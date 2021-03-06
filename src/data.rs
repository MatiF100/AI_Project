use ndarray::{arr2, Array2};

#[derive(Debug, Default, Clone)]
pub struct Animal {
    name: String,
    hair: f64,
    feathers: f64,
    eggs: f64,
    milk: f64,
    airborne: f64,
    aqatic: f64,
    predator: f64,
    toothed: f64,
    backbone: f64,
    breathes: f64,
    venomous: f64,
    fins: f64,
    legs: f64,
    tail: f64,
    domestic: f64,
    catsize: f64,
    ani_type: u8,
}

impl Animal {
    #[allow(dead_code)]
    pub fn even_list(mut vector: Vec<Self>) -> Vec<Self> {
        let mut classes = [0; 7];
        let limit = 9;

        vector = vector
            .into_iter()
            .filter(|a| {
                classes[a.ani_type as usize] += 1;
                classes[a.ani_type as usize] <= limit
            })
            .collect::<Vec<_>>();
        vector
    }
    pub fn partitioned_list(vector: Vec<Self>, training_ratio: f64) -> (Vec<Self>, Vec<Self>) {
        let mut classes: [usize; 7] = [0; 7];
        vector
            .iter()
            .for_each(|v| classes[v.ani_type as usize] += 1);
        let mut limits: [usize; 7] = classes
            .iter()
            .map(|&count| (count as f64 * training_ratio).round() as usize)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap();

        let mut train_v: Vec<Self> = Vec::with_capacity(limits.iter().sum());
        let mut ver_v: Vec<Self> = Vec::with_capacity(vector.len() - limits.iter().sum::<usize>());

        vector.into_iter().for_each(|val| {
            let idx = val.ani_type as usize;
            if limits[idx] > 0 {
                limits[idx] -= 1;
                train_v.push(val);
            } else {
                ver_v.push(val);
            }
        });

        (train_v, ver_v)
    }

    pub fn new_list(dataset: &str) -> Vec<Animal> {
        let mut values = dataset
            .lines()
            .map(|l| l.split(',').collect::<Vec<&str>>())
            .map(|v| Self::from_str(v))
            .collect::<Vec<_>>();
        //println!("{:?}", values[1]);
        //

        //values = Self::even_list(values);

        let max = values.iter().map(|a| a.legs as u8).max().unwrap();
        let min = values.iter().map(|a| a.legs as u8).min().unwrap();

        values
            .iter_mut()
            .for_each(|a| a.legs = (a.legs as u8 - min) as f64 / (max - min) as f64);

        values
    }
    fn from_str(data: Vec<&str>) -> Self {
        let mut an: Self = Default::default();
        let mut it = data.iter();
        an.name = it.next().unwrap().to_string();
        an.hair = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.feathers = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.eggs = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.milk = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.airborne = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.aqatic = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.predator = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.toothed = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.backbone = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.breathes = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.venomous = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.fins = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.legs = it.next().unwrap().parse::<f64>().unwrap();
        an.tail = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.domestic = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.catsize = match *it.next().unwrap() {
            "0" => 0.0,
            _ => 1.0,
        };
        an.ani_type = it.next().unwrap().parse::<u8>().unwrap() - 1;

        an
    }
    pub fn into_training_arr2(&self) -> (Array2<f64>, Array2<f64>) {
        (
            arr2(&[[
                self.hair,
                self.feathers,
                self.eggs,
                self.milk,
                self.airborne,
                self.aqatic,
                self.predator,
                self.toothed,
                self.backbone,
                self.breathes,
                self.venomous,
                self.fins,
                self.legs as f64,
                self.tail,
                self.domestic,
                self.catsize,
            ]])
            .t()
            .to_owned(),
            Array2::from_shape_vec(
                (7, 1),
                (0..7)
                    .map(|x| if x == self.ani_type { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            )
            .unwrap(),
        )
    }
}

/*
fn main() {
    let data = std::fs::read_to_string("zoo.data");
    Animal::new_list(&data.unwrap());
}
*/
