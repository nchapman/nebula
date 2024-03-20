use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
};

const CUDA_LINUX_GLOBS: &'static [&'static str] = &[
    "/usr/local/cuda/lib64/libnvidia-ml.so*",
    "/usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-ml.so*",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*",
    "/usr/lib/wsl/lib/libnvidia-ml.so*",
    "/usr/lib/wsl/drivers/*/libnvidia-ml.so*",
    "/opt/cuda/lib64/libnvidia-ml.so*",
    "/usr/lib*/libnvidia-ml.so*",
    "/usr/local/lib*/libnvidia-ml.so*",
    "/usr/lib/aarch64-linux-gnu/nvidia/current/libnvidia-ml.so*",
    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so*",
    "/opt/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so*",
];

const CUDA_WINDOWS_GLOBS: &'static [&'static str] = &["c:\\Windows\\System32\\nvml.dll"];

fn find_libs(system_patterns: Vec<String>, patterns: &[&str]) -> Result<HashSet<PathBuf>, String> {
    system_patterns
        .iter()
        .map(|s| s.to_string())
        .chain(patterns.iter().map(|s| s.to_string()))
        .try_fold(HashSet::new(), |mut acc, s| {
            for entry in glob::glob(&s).map_err(|e| e.to_string())? {
                let ee = entry.map_err(|e| e.to_string())?;
                if let Ok(ll) = std::fs::read_link(ee.clone()) {
                    acc.insert(ll);
                } else {
                    acc.insert(ee);
                }
            }
            if acc.len() > 0 {
                Ok(acc)
            } else {
                Err("no libs".to_string())
            }
        })
}

fn find_cuda_libs() -> Result<HashSet<PathBuf>, String> {
    if cfg!(target_os = "windows") {
        let ppaterns = env::var("PATH")
            .unwrap_or("".to_string())
            .split(";")
            .map(|s| format!("{}/{}*", s, "nvml.dll"))
            .collect::<Vec<String>>();
        find_libs(ppaterns, CUDA_WINDOWS_GLOBS)
    } else if cfg!(target_os = "linux") {
        let ppaterns = env::var("LD_LIBRARY_PATH")
            .unwrap_or("".to_string())
            .split(";")
            .map(|s| format!("{}/{}*", s, "libnvidia-ml.so"))
            .collect::<Vec<String>>();
        find_libs(ppaterns, CUDA_LINUX_GLOBS)
    } else {
        Err("no cuda libs".to_string())
    }
}

fn find_hip_libs() -> Result<HashSet<PathBuf>, String> {
    if cfg!(target_os = "windows") {
        let ppaterns = env::var("PATH")
            .unwrap_or("".to_string())
            .split(";")
            .map(|s| format!("{}/{}*", s, "amdhip64.dll"))
            .collect::<Vec<String>>();
        find_libs(ppaterns, &[])
    } else if cfg!(target_os = "linux") {
        if Path::new("/sys/module/amdgpu/version").exists() {
            let mut ss = HashSet::new();
            ss.insert(PathBuf::from("/sys/module/amdgpu/version"));
            Ok(ss)
        } else {
            Err("no libs".to_string())
        }
    } else {
        Err("no libs".to_string())
    }
}

// https://github.com/ggerganov/llama.cpp/blob/a836c8f534ab789b02da149fbdaf7735500bff74/Makefile#L364-L368
fn ggml_cudablas_build(llama_cpp: &mut cc::Build, ggml: &mut cc::Build) -> cc::Build {
    let mut ggml_cuda = cc::Build::new();
    for lib in [
        "cuda", "cublas", "culibos", "cudart", "cublasLt", "pthread", "dl", "rt",
    ] {
        println!("cargo:rustc-link-lib={}", lib);
    }
    if !ggml_cuda.get_compiler().is_like_msvc() {
        for lib in ["culibos", "pthread", "dl", "rt"] {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    if cfg!(target_arch = "aarch64") {
        ggml_cuda
            .flag_if_supported("-mfp16-format=ieee")
            .flag_if_supported("-mno-unaligned-access");
        ggml.flag_if_supported("-mfp16-format=ieee")
            .flag_if_supported("-mno-unaligned-access");
        llama_cpp
            .flag_if_supported("-mfp16-format=ieee")
            .flag_if_supported("-mno-unaligned-access");
    }

    ggml_cuda
        .cuda(true)
        .flag("-arch=all")
        .file("llama.cpp/ggml-cuda.cu")
        .include("llama.cpp");

    if ggml_cuda.get_compiler().is_like_msvc() {
        ggml_cuda.std("c++14");
    } else {
        ggml_cuda.flag("-std=c++11").std("c++11");
    }

    ggml.define("GGML_USE_CUBLAS", None);
    ggml_cuda.define("GGML_USE_CUBLAS", None);
    llama_cpp.define("GGML_USE_CUBLAS", None);
    ggml_cuda
}

fn ggml_hipblas_build(
    llama_cpp: &mut cc::Build,
    ggml: &mut cc::Build,
    hip_uma: bool,
    cuda_force_dmmv: bool,
) -> cc::Build {
    let mut ggml_hip = cc::Build::new();
    let rocm_path = if Path::new("/opt/rocm").exists() {
        "/opt/rocm"
    } else {
        "/usr"
    };

    println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    for lib in ["hipblas", "amdhip64", "rocblas"] {
        println!("cargo:rustc-link-lib={}", lib);
    }

    ggml_hip
        .compiler(&format!("{}/bin/hipcc", rocm_path))
        .flag("-arch=all")
        .file("llama.cpp/ggml-cuda.cu")
        .include("llama.cpp");

    if ggml_hip.get_compiler().is_like_msvc() {
        ggml_hip.std("c++14");
    } else {
        ggml_hip.flag("-std=c++11").std("c++11");
    }

    for gg in [&mut ggml_hip, ggml, llama_cpp] {
        gg.define("GGML_USE_CUBLAS", None);
        gg.define("GGML_USE_HIPBLAS", None);
        gg.define("GGML_CUDA_DMMV_X", Some("32"));
        gg.define("GGML_CUDA_MMV_Y", Some("1"));
        gg.define("K_QUANTS_PER_ITERATION", Some("2"));
        if hip_uma {
            gg.define("GGML_HIP_UMA", None);
        }
        if cuda_force_dmmv {
            gg.define("GGML_CUDA_FORCE_DMMV", None);
        }
    }
    ggml_hip
}

fn ggml_clblas_build(llama_cpp: &mut cc::Build, ggml: &mut cc::Build) -> cc::Build {
    let mut ggml_opencl = cc::Build::new();
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=clblast");
        println!("cargo:rustc-link-lib=framework=OpenCL");
    } else {
        println!("cargo:rustc-link-lib=clblast");
        println!("cargo:rustc-link-lib=OpenCL");
    }

    ggml_opencl
        .file("llama.cpp/ggml-opencl.cpp")
        .include("llama.cpp");

    if ggml_opencl.get_compiler().is_like_msvc() {
        ggml_opencl.std("c++14");
    } else {
        ggml_opencl.flag("-std=c++11").std("c++11");
    }

    ggml.define("GGML_USE_CLBLAST", None);
    ggml_opencl.define("GGML_USE_CLBLAST", None);
    llama_cpp.define("GGML_USE_CLBLAST", None);
    ggml_opencl
}

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    let mut ggml = cc::Build::new();
    let mut llama_cpp = cc::Build::new();

    ggml.cpp(false);
    llama_cpp.cpp(true);

    let cublas = env::var("CARGO_FEATURE_CUBLAS").is_ok();
    let openblas = env::var("CARGO_FEATURE_OPENBLAS").is_ok();
    let hipblas = env::var("CARGO_FEATURE_HIPBLAS").is_ok();
    let clblas = env::var("CARGO_FEATURE_CLBLAS").is_ok();
    let blis = env::var("CARGO_FEATURE_BLIS").is_ok();

    let mut ggml_acc = match (cublas, openblas, hipblas, clblas, blis) {
        (true, false, false, false, false) => Some(ggml_cudablas_build(&mut llama_cpp, &mut ggml)),
        (false, true, false, false, false) => {
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread/");
            println!("cargo:rustc-link-lib=openblas");
            ggml.define("GGML_USE_OPENBLAS", None);
            llama_cpp.define("GGML_USE_OPENBLAS", None);
            None
        }
        (false, false, true, false, false) => {
            let hip_uma = env::var("CARGO_FEATURE_HIP_UMA").is_ok();
            let cuda_force_dmmv = env::var("CARGO_FEATURE_CUDA_FORCE_DMMV").is_ok();
            Some(ggml_hipblas_build(
                &mut llama_cpp,
                &mut ggml,
                hip_uma,
                cuda_force_dmmv,
            ))
        }
        (false, false, false, true, false) => Some(ggml_clblas_build(&mut llama_cpp, &mut ggml)),
        (false, false, false, false, true) => {
            println!("cargo:rustc-link-lib=blis");
            ggml.define("GGML_USE_OPENBLAS", None);
            llama_cpp.define("GGML_USE_OPENBLAS", None);
            None
        }
        (false, false, false, false, false) => {
            if let Ok(_s) = find_cuda_libs() {
                Some(ggml_cudablas_build(&mut llama_cpp, &mut ggml))
            } else if let Ok(_s) = find_hip_libs() {
                let hip_uma = env::var("CARGO_FEATURE_HIP_UMA").is_ok();
                let cuda_force_dmmv = env::var("CARGO_FEATURE_CUDA_FORCE_DMMV").is_ok();
                Some(ggml_hipblas_build(
                    &mut llama_cpp,
                    &mut ggml,
                    hip_uma,
                    cuda_force_dmmv,
                ))
            } else {
                None
            }
        }
        _ => panic!("should be selected only 1 feature"),
    };

    for build in [&mut ggml, &mut llama_cpp] {
        let compiler = build.get_compiler();

        if cfg!(target_arch = "i686") || cfg!(target_arch = "x86_64") {
            let features = x86::Features::get_target();
            if compiler.is_like_clang() || compiler.is_like_gnu() {
                build.flag("-pthread");

                if features.avx {
                    build.flag("-mavx");
                }
                if features.avx2 {
                    build.flag("-mavx2");
                }
                if features.fma {
                    build.flag("-mfma");
                }
                if features.f16c {
                    build.flag("-mf16c");
                }
                if features.sse3 {
                    build.flag("-msse3");
                }
            } else if compiler.is_like_msvc() {
                match (features.avx2, features.avx) {
                    (true, _) => {
                        build.flag("/arch:AVX2");
                    }
                    (_, true) => {
                        build.flag("/arch:AVX");
                    }
                    _ => {}
                }
            }
        } else if cfg!(target_arch = "aarch64")
            && (compiler.is_like_clang() || compiler.is_like_gnu())
        {
            if cfg!(target_os = "macos") {
                build.flag("-mcpu=apple-m1");
            } else if env::var("HOST") == env::var("TARGET") {
                build.flag("-mcpu=native");
            }
            build.flag("-pthread");
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/191221178f51b6e81122c5bda0fd79620e547d07/Makefile#L133-L141
    if cfg!(target_os = "macos") {
        assert!(!cublas, "CUBLAS is not supported on macOS");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalKit");

        llama_cpp.define("_DARWIN_C_SOURCE", None);

        // https://github.com/ggerganov/llama.cpp/blob/3c0d25c4756742ebf15ad44700fabc0700c638bd/Makefile#L340-L343
        llama_cpp.define("GGML_USE_METAL", None);
        llama_cpp.define("GGML_USE_ACCELERATE", None);
        llama_cpp.define("ACCELERATE_NEW_LAPACK", None);
        llama_cpp.define("ACCELERATE_LAPACK_ILP64", None);
        println!("cargo:rustc-link-arg=framework=Accelerate");

        metal_hack(&mut ggml);
        ggml.include("./llama.cpp/ggml-metal.h");
    }

    if cfg!(target_os = "dragonfly") {
        llama_cpp.define("__BSD_VISIBLE", None);
    }

    if cfg!(target_os = "linux") {
        ggml.define("_GNU_SOURCE", None);
    }

    ggml.std("c11")
        .include("./llama.cpp")
        .file("llama.cpp/ggml.c")
        .file("llama.cpp/ggml-alloc.c")
        .file("llama.cpp/ggml-backend.c")
        .file("llama.cpp/ggml-quants.c")
        .define("GGML_USE_K_QUANTS", None);

    llama_cpp
        .define("_XOPEN_SOURCE", Some("600"))
        .include("llama.cpp")
        .std("c++11")
        .file("llama.cpp/llama.cpp")
        .file("llama.cpp/unicode.cpp");

    // Remove debug log output from `llama.cpp`
    let is_release = env::var("PROFILE").unwrap() == "release";
    if is_release {
        ggml.define("NDEBUG", None);
        llama_cpp.define("NDEBUG", None);
        if let Some(ggml_acc) = ggml_acc.as_mut() {
            ggml_acc.define("NDEBUG", None);
        }
    }

    if let Some(ggml_acc) = ggml_acc {
        println!("compiling ggml-acc");
        ggml_acc.compile("ggml-acc");
        println!("compiled ggml-acc");
    }

    println!("compiling ggml");
    ggml.compile("ggml");
    println!("compiled ggml");

    println!("compiling llama");
    llama_cpp.compile("llama");
    println!("compiled llama");

    let header = "llama.cpp/llama.h";

    println!("cargo:rerun-if-changed={header}");

    let bindings = bindgen::builder()
        .header(header)
        .derive_partialeq(true)
        .no_debug("llama_grammar_element")
        .prepend_enum_name(false)
        .derive_eq(true)
        .generate()
        .expect("failed to generate bindings for llama.cpp");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings to file");
}

// courtesy of https://github.com/rustformers/llm
fn metal_hack(build: &mut cc::Build) {
    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not defined"));

    let ggml_metal_path = {
        let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
            .expect("Could not read ggml-metal.metal")
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\"', "\\\"");

        let ggml_metal =
            std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

        let needle = r#"NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];"#;
        if !ggml_metal.contains(needle) {
            panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated.");
        }

        // Replace the runtime read of the file with a compile-time string
        let ggml_metal = ggml_metal.replace(
            needle,
            &format!(r#"NSString * src  = @"{ggml_metal_metal}";"#),
        );

        let patched_ggml_metal_path = out_dir.join("ggml-metal.m");
        std::fs::write(&patched_ggml_metal_path, ggml_metal)
            .expect("Could not write temporary patched ggml-metal.m");

        patched_ggml_metal_path
    };

    build.file(ggml_metal_path);
}

// Courtesy of https://github.com/rustformers/llm
fn get_supported_target_features() -> std::collections::HashSet<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(ToString::to_string)
        .collect()
}

mod x86 {
    #[allow(clippy::struct_excessive_bools)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Features {
        pub fma: bool,
        pub avx: bool,
        pub avx2: bool,
        pub f16c: bool,
        pub sse3: bool,
    }
    impl Features {
        pub fn get_target() -> Self {
            let features = crate::get_supported_target_features();
            Self {
                fma: features.contains("fma"),
                avx: features.contains("avx"),
                avx2: features.contains("avx2"),
                f16c: features.contains("f16c"),
                sse3: features.contains("sse3"),
            }
        }
    }
}
