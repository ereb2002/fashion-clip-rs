// src/embed.rs
use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array, Axis, IxDyn, s};
use ort::{
    environment::Environment,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
    SessionBuilder,
};
use std::path::Path;
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
        let environment = Environment::builder()
            .with_name("fashion-clip-text")
            .build()?
            .into_arc();

        // Crear sesión con la nueva API
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
            max_length: 77, // Longitud típica para CLIP
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

        // Preparar inputs como tensores con forma [1, max_length]
        let input_ids_array = Array::from_shape_vec((1, self.max_length), input_ids)?;
        let attention_mask_array = Array::from_shape_vec((1, self.max_length), attention_mask)?;

        // Convertir a tensores usando la nueva API
        let input_ids_tensor = Value::from_array(
            self.session.inputs[0].input_type.data_type()?,
            input_ids_array
        )?;
        
        let attention_mask_tensor = Value::from_array(
            self.session.inputs[1].input_type.data_type()?,
            attention_mask_array
        )?;

        // Ejecutar sesión
        let outputs = self.session.run(vec![input_ids_tensor, attention_mask_tensor])?;

        // Extraer embeddings (mean pooling)
        let token_embeddings = outputs[0].try_extract::<f32>()?;
        
        // Mean pooling (promediar sobre la dimensión de tokens)
        let embedding = token_embeddings.view()
            .mean_axis(Axis(1))
            .ok_or_else(|| anyhow!("Mean pooling failed"))?
            .iter()
            .copied()
            .collect();

        Ok(embedding)
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
        let environment = Environment::builder()
            .with_name("fashion-clip-image")
            .build()?
            .into_arc();

        // Crear sesión con la nueva API
        let session = SessionBuilder::new()
            .with_environment(environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            image_size: (224, 224), // Tamaño estándar para ViT
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        })
    }

    pub fn encode(&self, image_bytes: &[u8]) -> Result<Vec<f32>> {
        // Cargar y preprocesar imagen
        let img = image::load_from_memory(image_bytes)?;
        let tensor = self.preprocess_image(&img)?;
        
        // Ejecutar sesión
        let outputs = self.session.run(vec![tensor])?;

        // Extraer embeddings (CLS token o pooled output)
        let embedding = outputs[0]
            .try_extract::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();

        Ok(embedding)
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<Value> {
        // Redimensionar
        let resized = img.resize_exact(
            self.image_size.0,
            self.image_size.1,
            image::imageops::FilterType::CatmullRom,
        );

        let rgb = resized.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        // Crear array con forma [1, 3, height, width]
        let mut array = Array::zeros((1, 3, height as usize, width as usize));

        // Normalizar píxeles
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    array[[0, c, y as usize, x as usize]] = 
                        (pixel[c] as f32 / 255.0 - self.mean[c]) / self.std[c];
                }
            }
        }

        Ok(Value::from_array(
            self.session.inputs[0].input_type.data_type()?,
            array
        )?)
    }
}

// Función auxiliar para crear sesión (para compatibilidad con código existente)
pub fn create_session(model_path: &str) -> Result<Session> {
    let environment = Environment::builder()
        .with_name("embed-rs")
        .build()?
        .into_arc();
    
    let num_cpus = num_cpus::get();
    let session = SessionBuilder::new()
        .with_environment(environment)?
        .with_parallel_execution(true)?
        .with_intra_threads(num_cpus as i16)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;
    
    Ok(session)
}
