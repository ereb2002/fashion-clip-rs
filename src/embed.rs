// src/embed.rs - Versión ultra simplificada
use anyhow::Result;
use ort::{
    environment::Environment,
    session::{builder::GraphOptimizationLevel, Session},
    session::builder::SessionBuilder,
    value::{Tensor, Value},
};
use std::sync::Arc;

pub fn create_session(model_path: &str) -> Result<Session> {
    let environment = Arc::new(
        Environment::builder()
            .with_name("embed-rs")
            .build()?
    );

    let session = SessionBuilder::new()
        .with_environment(environment)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    
    Ok(session)
}

pub fn encode_dummy(session: &Session) -> Result<Vec<f32>> {
    // Crear un tensor dummy
    let dummy_data = vec![0.0f32; 512];
    let tensor = Tensor::from_array(
        session.inputs[0].input_type.data_type()?,
        &dummy_data,
        &[1, 512]
    )?;
    
    let outputs = session.run(vec![
        Value::from_tensor(tensor)?
    ])?;
    
    let output = outputs[0].try_extract_tensor::<f32>()?;
    Ok(output.as_slice().unwrap().to_vec())
}
