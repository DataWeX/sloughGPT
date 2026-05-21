pub mod inference {
    pub mod request {
        include!("generated/inference.rs");
    }
    pub mod response {
        include!("generated/inference.rs");
    }
}

pub mod training {
    pub mod request {
        include!("generated/training.rs");
    }
    pub mod response {
        include!("generated/training.rs");
    }
}
