use std::{
    collections::HashSet,
    env,
    fs::{read_dir, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use cc::Build;

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

fn compile_opencl(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_CLBLAST");
    cxx.flag("-DGGML_USE_CLBLAST");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    }

    cxx.file("./llama.cpp/ggml-opencl.cpp");
}

fn compile_openblas(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/openblas")
        .include("/usr/local/include/openblas");
    println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/blis")
        .include("/usr/local/include/blis");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=blis");
}

fn compile_cuda(cxx_flags: &str, outdir: &PathBuf) {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");

    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_default();
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            cuda_path
        );
    }

    let libs = if cfg!(target_os = "linux") {
        "cuda culibos cublas cudart cublasLt pthread dl rt"
    } else {
        "cuda cublas cudart cublasLt"
    };

    for lib in libs.split_whitespace() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut nvcc = cc::Build::new();

    let env_flags = vec![
        ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
        ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
        ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    ];

    let nvcc_flags = "--forward-unknown-to-host-compiler -arch=native ";

    for nvcc_flag in nvcc_flags.split_whitespace() {
        nvcc.flag(nvcc_flag);
    }

    for cxx_flag in cxx_flags.split_whitespace() {
        nvcc.flag(cxx_flag);
    }

    for env_flag in env_flags {
        let mut flag_split = env_flag.0.split("=");
        if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
            nvcc.flag(&format!("{}={}", env_flag.1, val));
        } else {
            nvcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
        }
    }

    if cfg!(target_os = "linux") {
        nvcc.compiler("nvcc")
            .file("./llama.cpp/ggml-cuda.cu")
            .flag("-Wno-pedantic")
            .include("./llama.cpp/ggml-cuda.h")
            .compile("ggml-cuda");
    } else {
        let include_path = format!("{}\\include", cuda_path);

        let object_file = outdir
            .join("llama.cpp")
            .join("ggml-cuda.o")
            .to_str()
            .expect("Could not build ggml-cuda.o filename")
            .to_string();

        std::process::Command::new("nvcc")
            .arg("-ccbin")
            .arg(
                cc::Build::new()
                    .get_compiler()
                    .path()
                    .parent()
                    .unwrap()
                    .join("cl.exe"),
            )
            .arg("-I")
            .arg(&include_path)
            .arg("-o")
            .arg(&object_file)
            .arg("-x")
            .arg("cu")
            .arg("-maxrregcount=0")
            .arg("--machine")
            .arg("64")
            .arg("--compile")
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
            .arg("-D_WINDOWS")
            .arg("-DNDEBUG")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-D_CRT_SECURE_NO_WARNINGS")
            .arg("-D_MBCS")
            .arg("-DWIN32")
            .arg(r"-Illama.cpp\include\ggml")
            .arg(r"llama.cpp\ggml-cuda.cu")
            .status()
            .unwrap();

        nvcc.object(&object_file);
        nvcc.flag("-DGGML_USE_CUBLAS");
        nvcc.include(&include_path);
    }
}

fn compile_ggml(cx: &mut Build, cx_flags: &str) {
    for cx_flag in cx_flags.split_whitespace() {
        cx.flag(cx_flag);
    }

    cx.include("./llama.cpp")
        .file("./llama.cpp/ggml.c")
        .file("./llama.cpp/ggml-alloc.c")
        .file("./llama.cpp/ggml-backend.c")
        .file("./llama.cpp/ggml-quants.c")
        .cpp(false)
        .define("_GNU_SOURCE", None)
        .define("GGML_USE_K_QUANTS", None)
        .compile("ggml");
}

fn compile_metal(cx: &mut Build, cxx: &mut Build, out: &PathBuf) {
    cx.flag("-DGGML_USE_METAL")
        .flag("-DGGML_METAL_NDEBUG")
        .flag("-DGGML_METAL_EMBED_LIBRARY");
    cxx.flag("-DGGML_USE_METAL")
        .flag("-DGGML_METAL_EMBED_LIBRARY");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";
    println!("cargo:rerun-if-changed={GGML_METAL_METAL_PATH}");
    println!("cargo:rerun-if-changed={GGML_METAL_PATH}");

    let metal_embed_asm = out.join("ggml-metal-embed.s");
    let metal_source_embed = out.join("ggml-metal-embed.metal");

    let mut common_h = String::new();
    let mut metal_source = String::new();
    File::open("llama.cpp/ggml-common.h")
        .unwrap()
        .read_to_string(&mut common_h)
        .unwrap();
    File::open("llama.cpp/ggml-metal.metal")
        .unwrap()
        .read_to_string(&mut metal_source)
        .unwrap();

    let mut embedded_metal = String::new();
    embedded_metal.push_str(&metal_source.replace("#include \"ggml-common.h\"", common_h.as_str()));

    let mut embed_metal_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&metal_source_embed)
        .unwrap();
    embed_metal_file
        .write_all(embedded_metal.as_bytes())
        .unwrap();

    let mut embed_asm_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&metal_embed_asm)
        .unwrap();
    embed_asm_file
        .write_all(
            b"\
            .section __DATA,__ggml_metallib\n
            .globl _ggml_metallib_start\n
            _ggml_metallib_start:\n
            .incbin \"",
        )
        .unwrap();
    embed_asm_file
        .write_all(metal_source_embed.into_os_string().as_encoded_bytes())
        .unwrap();
    embed_asm_file
        .write_all(
            b"\"\n
            .globl _ggml_metallib_end\n
            _ggml_metallib_end:\n
        ",
        )
        .unwrap();

    cx.file(GGML_METAL_PATH);
    cx.file(metal_embed_asm)
        .flag("-c")
        .compile("ggml-metal-embed.o");
}

fn compile_llama(cxx: &mut Build, cxx_flags: &str, out_path: &PathBuf, ggml_type: &str) {
    for cxx_flag in cxx_flags.split_whitespace() {
        cxx.flag(cxx_flag);
    }

    //    let ggml_obj = PathBuf::from(&out_path).join("llama.cpp/ggml.o");
    //    cxx.object(ggml_obj);

    if !ggml_type.is_empty() {
        if ggml_type.eq("metal") {
            for fs_entry in read_dir(out_path).unwrap() {
                let fs_entry = fs_entry.unwrap();
                let path = fs_entry.path();
                if path.ends_with("-metal-embed.o.a") {
                    cxx.object(path);
                    break;
                }
            }
        } else {
            let ggml_feature_obj =
                PathBuf::from(&out_path).join(format!("llama.cpp/ggml-{}.o", ggml_type));
            cxx.object(ggml_feature_obj);
        }
    }

    if cfg!(target_os = "windows") {
        let build_info_str = std::process::Command::new("sh")
            .arg("llama.cpp/scripts/build-info.sh")
            .output()
            .expect("Failed to generate llama.cpp/common/build-info.cpp from the shell script.");

        let mut build_info_file = File::create("llama.cpp/common/build-info.cpp")
            .expect("Could not create llama.cpp/common/build-info.cpp file");
        std::io::Write::write_all(&mut build_info_file, &build_info_str.stdout)
            .expect("Could not write to llama.cpp/common/build-info.cpp file");

        cxx.shared_flag(true)
            .file("./llama.cpp/common/build-info.cpp");
    }

    const LLAMACPP_PATH: &str = "llama.cpp/llama.cpp";
    const PATCHED_LLAMACPP_PATH: &str = "llama.cpp/llama-patched.cpp";
    let llamacpp_code =
        std::fs::read_to_string(LLAMACPP_PATH).expect("Could not read llama.cpp source file.");
    let needle1 =
        r#"#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)"#;
    let needle2 =
        r#"#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)"#;
    let needle3 =
        r#"#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)"#;
    if !llamacpp_code.contains(needle1)
        || !llamacpp_code.contains(needle2)
        || !llamacpp_code.contains(needle3)
    {
        panic!("llama.cpp does not contain the needles to be replaced; the patching logic needs to be reinvestigated!");
    }
    let patched_llamacpp_code = llamacpp_code
        .replace(
            needle1,
            "#include \"log.h\"\n#define LLAMA_LOG_INFO(...)  LOG(__VA_ARGS__)",
        )
        .replace(needle2, "#define LLAMA_LOG_WARN(...)  LOG(__VA_ARGS__)")
        .replace(needle3, "#define LLAMA_LOG_ERROR(...) LOG(__VA_ARGS__)");
    std::fs::write(&PATCHED_LLAMACPP_PATH, patched_llamacpp_code)
        .expect("Attempted to write the patched llama.cpp file out to llama-patched.cpp");

    cxx.shared_flag(true)
        .file("./llama.cpp/common/common.cpp")
        .file("./llama.cpp/unicode.cpp")
        .file("./llama.cpp/common/sampling.cpp")
        .file("./llama.cpp/common/grammar-parser.cpp")
        .file("./llama.cpp/llama-patched.cpp")
        .cpp(true)
        .compile("binding");
}

fn main() {
    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }
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

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    let mut cx_flags = String::from("");
    let mut cxx_flags = String::from("");

    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        cx_flags.push_str(" -std=c11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -pthread -march=native -mtune=native");
        cxx_flags.push_str(" -std=c++11 -Wall -Wdeprecated-declarations -Wunused-but-set-variable -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -fPIC -pthread -march=native -mtune=native");
    } else if cfg!(target_os = "windows") {
        cx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
        cxx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
    }

    let mut cx = cc::Build::new();
    let mut cxx = cc::Build::new();
    let mut ggml_type = String::new();

    cxx.include("./llama.cpp/common")
        .include("./llama.cpp")
        .include("./include_shims");

    if cfg!(feature = "opencl") {
        compile_opencl(&mut cx, &mut cxx);
        ggml_type = "opencl".to_string();
    } else if cfg!(feature = "openblas") {
        compile_openblas(&mut cx);
    } else if cfg!(feature = "blis") {
        compile_blis(&mut cx);
    } else if !cfg!(feature = "no-metal") && cfg!(target_os = "macos") {
        compile_metal(&mut cx, &mut cxx, &out_path);
        ggml_type = "metal".to_string();
    }

    if cfg!(feature = "no-metal") && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        cx.define("GGML_USE_ACCELERATE", None);
    }

    if cfg!(feature = "cuda") {
        cx_flags.push_str(" -DGGML_USE_CUBLAS");
        cxx_flags.push_str(" -DGGML_USE_CUBLAS");

        if cfg!(target_os = "linux") {
            cx.include("/usr/local/cuda/include")
                .include("/opt/cuda/include");
            cxx.include("/usr/local/cuda/include")
                .include("/opt/cuda/include");

            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                cx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
                cxx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
            }
        } else {
            cx.flag("/MT");
            cxx.flag("/MT");
        }

        compile_ggml(&mut cx, &cx_flags);

        compile_cuda(&cxx_flags, &out_path);

        if !cfg!(feature = "logfile") {
            cxx.define("LOG_DISABLE_LOGS", None);
        }
        compile_llama(&mut cxx, &cxx_flags, &out_path, "cuda");
    } else {
        compile_ggml(&mut cx, &cx_flags);

        if !cfg!(feature = "logfile") {
            cxx.define("LOG_DISABLE_LOGS", None);
        }
        compile_llama(&mut cxx, &cxx_flags, &out_path, &ggml_type);
    }
}
