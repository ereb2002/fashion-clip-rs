// src/embed.rs
use anyhow::{anyhow, Result};
use image::DynamicImage;
use ndarray::{Array, Axis};
use ort::{
    environment::Environment,
    session::{builder::GraphOptimizationLevel, Session},
    session::builder::SessionBuilder,
    value::{Tensor, Value},
};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct EmbedText {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl EmbedText {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        // Crear entorno
        let environment = Arc::new(
            Environment::builder()
                .with_name("fashion-clip-text")
                .build()?
        );

        // Crear sesión
        let session = SessionBuilder::new()
            .with_environment(environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Cargar tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Error loading tokenizer: {}", e))?;

        Ok(Self {
            session,
            tokenizer,
            max_length: 77,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenizar
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        // Truncar/padding a max_length
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
        } else {
            input_ids.resize(self.max_length, 0);
            attention_mask.resize(self.max_length, 0);
        }

        // Convertir a i64 (típico para modelos de texto)
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

        // Crear tensores
        let input_ids_tensor = Tensor::from_array(
            self.session.inputs[0].input_type.data_type()?,
            &input_ids_i64,
            &[1, self.max_length]
        )?;

        let attention_mask_tensor = Tensor::from_array(
            self.session.inputs[1].input_type.data_type()?,
            &attention_mask_i64,
            &[1, self.max_length]
        )?;

        // Ejecutar sesión
        let outputs = self.session.run(vec![
            Value::from_tensor(input_ids_tensor)?,
            Value::from_tensor(attention_mask_tensor)?,
        ])?;

        // Extraer embeddings
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let shape = output_tensor.shape();
        
        if shape.len() == 3 {
            // Forma [batch, seq_len, hidden_size]
            let batch_size = shape[0];
            let seq_len = shape[1];
            let hidden_size = shape[2];
            
            let output_array = Array::from_shape_vec(
                (batch_size, seq_len, hidden_size),
                output_tensor.as_slice().unwrap().to_vec()
            )?;
            
            // Mean pooling
            let pooled = output_array.mean_axis(ndarray::Axis(1))
                .ok_or_else(|| anyhow!("Mean pooling failed"))?;
            
            Ok(pooled.to_vec())
        } else {
            // Ya es pooled
            Ok(output_tensor.as_slice().unwrap().to_vec())
        }
    }
}

pub struct EmbedImage {
    session: Session,
    image_size: (u32, u32),
    mean: [f32; 3],
    std: [f32; 3],
}

impl EmbedImage {
    pub fn new(model_path: &str) -> Result<Self> {
        // Crear entorno
        let environment = Arc::new(
            Environment::builder()
                .with_name("fashion-clip-image")
                .build()?
        );

        // Crear sesión
        let session = SessionBuilder::new()
            .with_environment(environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            image_size: (224, 224),
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        })
    }

    pub fn encode(&self, image_bytes: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(image_bytes)?;
        let tensor = self.preprocess_image(&img)?;
        
        let outputs = self.session.run(vec![
            Value::from_tensor(tensor)?
        ])?;

        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        Ok(output_tensor.as_slice().unwrap().to_vec())
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<Tensor<f32>> {
        let resized = img.resize_exact(
            self.image_size.0,
            self.image_size.1,
            image::imageops::FilterType::CatmullRom,
        );

        let rgb = resized.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        // Crear array con forma [1, 3, height, width]
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb.get_pixel(x, y);
                    let normalized = (pixel[c] as f32 / 255.0 - self.mean[c]) / self.std[c];
                    data.push(normalized);
                }
            }
        }

        // Crear tensor
        let tensor = Tensor::from_array(
            self.session.inputs[0].input_type.data_type()?,
            &data,
            &[1, 3, height as usize, width as usize]
        )?;
        
        Ok(tensor)
    }
}

// Función auxiliar simplificada
pub fn create_session(model_path: &str) -> Result<Session> {
    let environment = Arc::new(
        Environment::builder()
            .with_name("embed-rs")
            .build()?
    );

    let session = SessionBuilder::new()
        .with_environment(environment)?
        .with_parallel_execution(true)?
        .with_intra_threads(num_cpus::get() as i16)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;
    
    Ok(session)
}
