use std::{
    env,
    path::{Path, PathBuf},
};

mod common {}

mod linux {}

mod macos {}

#[cfg(windows)]
mod windows {

    lazy_static::lazy_static! {
        static ref LLAMACPP_DIR: &'static str = "llama.cpp";
        static ref CMAKE_TARGETS: &'static[&'static str] = &["llama", "llava"];
        //TODO add debug variant
        static ref CMAKE_DEFS: std::collections::HashMap<&'static str, &'static str> = maplit::hashmap!{
            "BUILD_SHARED_LIBS" => "on",
            "LLAMA_NATIVE" => "off",
            "LLAMA_SERVER_VERBOSE" => "off",
            "CMAKE_BUILD_TYPE" => "Release"
        };
        static ref COMMON_CPU_DEFS: std::collections::HashMap<&'static str, &'static str> = maplit::hashmap!{
            "CMAKE_POSITION_INDEPENDENT_CODE" => "on"};
        static ref ARCH: String = std::env::consts::ARCH.to_string();
        static ref DIST_BASE: String = {
            let dist_base = format!("../dist/windows-{}", std::env::consts::ARCH);
            std::fs::create_dir_all(&dist_base).expect("can`t create dist directory");
            dist_base
        };

        static ref CUDA_DIR: (Option<String>, Option<String>) = {
            match std::env::var("CUDA_LIB_DIR"){
                Err(_) => {
                    match powershell_script::run("(get-command -ea 'silentlycontinue' nvcc).path"){
                        Ok(path) => {
                            let path = path.stdout().unwrap();
                            let mut lib_path = std::path::PathBuf::from(path.clone());
                            lib_path.pop();
                            let mut include_path = std::path::PathBuf::from(path);
                            include_path.pop();
                            include_path.pop();
                            (Some(lib_path.into_os_string().into_string().unwrap()), Some(include_path.into_os_string().into_string().unwrap()))
                        }
                        Err(_) => {
                            (None, None)
                        }
                    }
                }
                Ok(cuda_lib_dir) => {
                    (Some(cuda_lib_dir), None)
                }
            }
        };

        static ref DUMPBIN: Option<String> = match powershell_script::run("(get-command -ea 'silentlycontinue' dumpbin).path"){
            Ok(path) => {
                Some(path.stdout().unwrap())
            }
            Err(_) => {
                None
            }
        };

        static ref CMAKE_CUDA_ARCHITECTURES: String = {
            match std::env::var("CMAKE_CUDA_ARCHITECTURES"){
                Ok(cmake_cuda_architectures) => cmake_cuda_architectures,
                Err(_) => "50;52;61;70;75;80".to_string()
            }
        };
        static ref SIGN_TOOL: String = {
            std::env::var("SIGN_TOOL").unwrap_or("C:\\Program Files (x86)\\Windows Kits\\8.1\\bin\\x64\\signtool.exe".to_string())
        };
        static ref KEY_CONTAINER: Option<String> = None;
    }

    fn build(
        src_dir: &str,
        build_dir: &str,
        cmake_defs: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) {
        println!("build with: cmake -S {src_dir} -B {build_dir} {cmake_defs:?}");
        let mut dst = cmake::Config::new(src_dir);
        let mut dd = &mut dst;
        dd = dd.out_dir(build_dir);
        for (k, v) in cmake_defs.iter() {
            if v.is_empty() {
                dd = dd.build_arg(k);
            } else {
                dd = dd.define(k, v);
            }
        }
        for t in targets {
            dd = dd.target(t);
        }
        dd.build();
    }

    fn sign(build_dir: &str) {
        if let Some(kk) = &*KEY_CONTAINER {
            print!("Signing {build_dir}/bin/*.exe  {build_dir}/bin/*.dll");
            for entry in glob::glob(&format!("{build_dir}/bin/*.exe"))
                .expect("Failed to read glob pattern")
                .chain(
                    glob::glob(&format!("{build_dir}/bin/*.dll"))
                        .expect("Failed to read glob pattern"),
                )
            {
                if let Ok(path) = entry {
                    let path = path.into_os_string().into_string().unwrap();
                    powershell_script::run(&format!("{} sign /v /fd sha256 /t http://timestamp.digicert.com /f \"{kk}\" /csp \"Google Cloud KMS Provider\" /kc \"kk\" {path}", *SIGN_TOOL)).expect("sign error");
                }
            }
        }
    }

    fn install(build_dir: &str, dist_dir: &str) {
        println!("Installing binaries to dist dir {dist_dir}");
        std::fs::create_dir_all(dist_dir).expect("can`t create dist directory");
        for entry in glob::glob(&format!("{build_dir}/bin/*.exe"))
            .expect("Failed to read glob pattern")
            .chain(
                glob::glob(&format!("{build_dir}/bin/*.dll")).expect("Failed to read glob pattern"),
            )
        {
            if let Ok(path) = entry {
                let path = path.into_os_string().into_string().unwrap();
                powershell_script::run(&format!(
                    "copy-item -Path {path} -Destination {dist_dir} -Force"
                ))
                .expect("install error");
            }
        }
    }

    fn build_cpu(
        src_dir: &str,
        dist_dir: &str,
        cmake_defs: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) {
        let cmake_defs: std::collections::HashMap<&str, &str> = COMMON_CPU_DEFS
            .iter()
            .chain(
                maplit::hashmap! {
                    "LLAMA_AVX" => "off",
                    "LLAMA_AVX2" => "off",
                    "LLAMA_AVX512" => "off",
                    "LLAMA_FMA" => "off",
                    "LLAMA_F16C" => "off"
                }
                .iter()
                .chain(cmake_defs.iter()),
            )
            .map(|(k, v)| (*k, *v))
            .collect();
        println!("Building LCD CPU");
        let build_dir = format!("target/windows/{}/cpu", *ARCH);
        build(src_dir, &build_dir, &cmake_defs, targets);
        sign(&build_dir);
        install(&build_dir, &format!("{dist_dir}/cpu"));
    }

    fn build_cpu_avx(
        src_dir: &str,
        dist_dir: &str,
        cmake_defs: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) {
        let cmake_defs: std::collections::HashMap<&str, &str> = COMMON_CPU_DEFS
            .iter()
            .chain(
                maplit::hashmap! {
                    "LLAMA_AVX" => "on",
                    "LLAMA_AVX2" => "off",
                    "LLAMA_AVX512" => "off",
                    "LLAMA_FMA" => "off",
                    "LLAMA_F16C" => "off"
                }
                .iter()
                .chain(cmake_defs.iter()),
            )
            .map(|(k, v)| (*k, *v))
            .collect();
        println!("Building AVX CPU");
        let build_dir = format!("target/windows/{}/cpu_avx", *ARCH);
        build(src_dir, &build_dir, &cmake_defs, targets);
        sign(&build_dir);
        install(&build_dir, &format!("{dist_dir}/cpu_avx"));
    }

    fn build_cpu_avx2(
        src_dir: &str,
        dist_dir: &str,
        cmake_defs: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) {
        let cmake_defs: std::collections::HashMap<&str, &str> = COMMON_CPU_DEFS
            .iter()
            .chain(
                maplit::hashmap! {
                    "LLAMA_AVX" => "on",
                    "LLAMA_AVX2" => "on",
                    "LLAMA_AVX512" => "off",
                    "LLAMA_FMA" => "on",
                    "LLAMA_F16C" => "on"
                }
                .iter()
                .chain(cmake_defs.iter()),
            )
            .map(|(k, v)| (*k, *v))
            .collect();
        println!("Building AVX2 CPU");
        let build_dir = format!("target/windows/{}/cpu_avx2", *ARCH);
        build(src_dir, &build_dir, &cmake_defs, targets);
        sign(&build_dir);
        install(&build_dir, &format!("{dist_dir}/cpu_avx2"));
    }

    fn build_cuda(
        src_dir: &str,
        dist_dir: &str,
        _cmake_defs: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) {
        if let (Some(cuda_lib_dir), Some(cuda_include_dir)) = &*CUDA_DIR {
            let mut nvcc = std::path::PathBuf::from(cuda_lib_dir);
            nvcc.push("nvcc.exe");
            let nvcc = nvcc.into_os_string().into_string().unwrap();
            let cuda_version = powershell_script::run(&format!(
                "(get-item ({nvcc} | split-path | split-path)).Basename"
            ))
            .expect("can`t get cuda version");
            let build_dir = format!("target/windows/{}/cuda_{cuda_version}", *ARCH);
            let disst_dir = format!("{dist_dir}/cuda_{cuda_version}");
            let cmake_defs = maplit::hashmap! {
                "LLAMA_CUDA" => "ON",
                "LLAMA_AVX" => "on",
                "LLAMA_AVX2" => "off",
                "CUDAToolkit_INCLUDE_DIR" => &cuda_include_dir,
                "CMAKE_CUDA_FLAGS" => "-t8",
                "CMAKE_CUDA_ARCHITECTURES" => &*CMAKE_CUDA_ARCHITECTURES
            };
            println!("Building CUDA GPU");
            build(src_dir, &build_dir, &cmake_defs, targets);
            sign(&build_dir);
            install(&build_dir, &disst_dir);
            println!("copying CUDA dependencies to {dist_dir}-{}", *ARCH);
            for entry in glob::glob(&format!("{cuda_lib_dir}/cudart64_*.dll"))
                .expect("Failed to read glob pattern")
                .chain(
                    glob::glob(&format!("{cuda_lib_dir}/cublas64_*.dll"))
                        .expect("Failed to read glob pattern")
                        .chain(
                            glob::glob(&format!("{cuda_lib_dir}/cublasLt64_*.dll"))
                                .expect("Failed to read glob pattern"),
                        ),
                )
            {
                if let Ok(path) = entry {
                    let path = path.into_os_string().into_string().unwrap();
                    powershell_script::run(&format!("cp {path} {dist_dir}-{}\\", *ARCH))
                        .expect("sign error");
                }
            }
        }
    }

    pub fn bbuild() {
        build_cpu(
            *LLAMACPP_DIR,
            &format!("dist/windows/{}/", *ARCH),
            &*CMAKE_DEFS,
            *CMAKE_TARGETS,
        );

        if ::std::is_x86_feature_detected!("avx") {
            build_cpu_avx(
                *LLAMACPP_DIR,
                &format!("dist/windows/{}/", *ARCH),
                &*CMAKE_DEFS,
                *CMAKE_TARGETS,
            );
        }
        if ::std::is_x86_feature_detected!("avx2") {
            build_cpu_avx2(
                *LLAMACPP_DIR,
                &format!("dist/windows/{}/", *ARCH),
                &*CMAKE_DEFS,
                *CMAKE_TARGETS,
            );
        }
        build_cuda(
            *LLAMACPP_DIR,
            &format!("dist/windows/{}/", *ARCH),
            &*CMAKE_DEFS,
            *CMAKE_TARGETS,
        );
    }
}

fn main() {
    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    let bindings = bindgen::builder()
        .clang_args(&["-x", "c++", "-std=c++11", "-I./llama.cpp"])
        .header("llama.cpp/llama.h")
        .header("llama.cpp/examples/llava/clip.h")
        .header("llama.cpp/examples/llava/llava.h")
        //        .allowlist_function("llama_load_model_from_file")
        //        .allowlist_function("clip_model_load")
        //       .allowlist_item("LLAMA_TOKEN_TYPE_BYTE")
        .allowlist_type("llama_kv_cache_view")
        .allowlist_type("llama_batch")
        .allowlist_type("llama_seq_id")
        .allowlist_type("llava_image_embed")
        .allowlist_type("ggml_log_callback")
        .allowlist_type("ggml_numa_strategy")
        .allowlist_type("llama_timings")
        .allowlist_type("llama_context")
        .allowlist_type("llama_context_params")
        .allowlist_type("llama_token_data_array")
        .allowlist_type("llama_token")
        .allowlist_type("llama_vocab_type")
        .allowlist_type("llama_token_type")
        .allowlist_type("llama_model")
        .allowlist_type("llama_model_params")
        .allowlist_type("clip_ctx")
        .allowlist_type("llama_grammar_element")
        .allowlist_type("llama_image_embed")
        .allowlist_type("llama_grammar")
        .allowlist_type("ggml_log_level")
        .derive_partialeq(true)
        .no_debug("llama_grammar_element")
        .prepend_enum_name(false)
        .derive_eq(true)
        .generate()
        .expect("failed to generate bindings for llama.cpp");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings to file");

    #[cfg(windows)]
    windows::bbuild();
}
