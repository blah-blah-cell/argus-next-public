//! Appends a verified payload to the setup stub and reports the result.

use argus_setup::container::build_payload;
use argus_setup::sha256::{hex, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("Build failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut stub = None;
    let mut dist = None;
    let mut output = None;
    let mut version = None;

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--stub" => {
                stub = Some(PathBuf::from(
                    arguments.next().ok_or("--stub needs a path")?,
                ))
            }
            "--dist" => {
                dist = Some(PathBuf::from(
                    arguments.next().ok_or("--dist needs a path")?,
                ))
            }
            "--out" => output = Some(PathBuf::from(arguments.next().ok_or("--out needs a path")?)),
            "--version" => version = Some(arguments.next().ok_or("--version needs a value")?),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let stub = stub.ok_or("--stub is required")?;
    let dist = dist.ok_or("--dist is required")?;
    let output = output.ok_or("--out is required")?;
    let version = version.ok_or("--version is required")?;

    let count = build_payload(&dist, &version, &stub, &output)?;
    let bytes = fs::read(&output)?;
    println!("Setup: {}", output.display());
    println!("Payload entries: {count}");
    println!("Size: {} bytes", bytes.len());
    println!("SHA256: {}", hex(&Sha256::digest(&bytes)));
    Ok(())
}
