use std::fs::OpenOptions;
use std::io::Write;
use std::io::Result;
use std::process::Command;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(feature = "tts") {
        if cfg!(target_os = "windows") {}
        else if cfg!(target_os = "macos") {}
        else if cfg!(target_os = "linux") {    
            let distribution = get_linux_distribution();
            match distribution {
                Some(distro_string) => {
                    if distro_string == String::from("ubuntu") {
                        build_dependencies_for_ubuntu();
                        append_to_log("Successfully built dependencies for ubuntu").unwrap();
                    }
                },
                None => {}
            }
        }
        append_to_log("Running TTS example!").unwrap();
    } else {
        append_to_log("Running another example!").unwrap();
    }
}

fn get_linux_distribution() -> Option<String> {
    if let Ok(file) = File::open("/etc/os-release") {
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            if let Ok(line) = line {
                if line.starts_with("ID=") {
                    let id = line.trim_start_matches("ID=").replace("\"", "");
                    return Some(id);
                }
            }
        }
    }
    None
}

fn build_dependencies_for_ubuntu() -> () {
    Command::new("sh")
        .arg("-c")
        .arg("apt install clang")
        .status()
        .expect("Failed to build clang");
    Command::new("sh")
        .arg("-c")
        .arg("apt-get install espeak-ng")
        .status()
        .expect("Failed to build espeak-ng");
    let libtorch_build_prompt = r#"
    mkdir -p ./libtorch/ && 
    wget -O ./libtorch/libtorch-2.0.0.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip && 
    unzip ./libtorch/libtorch-2.0.0.zip && 
    export LIBTORCH=$(realpath ./libtorch/) && 
    export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
    "#;    
    Command::new("sh")
        .arg("-c")
        .arg(libtorch_build_prompt)
        .status()
        .expect("Failed to build libtorch");
}

fn append_to_log(content: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("nebula-build-log.txt")?;    
    writeln!(file, "{}", content)?;    
    Ok(())
}
