use std::{
    env,
    path::{Path, PathBuf},
};

//#[cfg(feature = "build")]
#[cfg(any(target_os = "macos", target_os = "unix"))]
mod common {
    lazy_static::lazy_static! {
        pub static ref LLAMACPP_DIR: &'static str = "llama.cpp";
        pub static ref CMAKE_TARGETS: &'static[&'static str] = &["llama", "llava_shared"];
        //TODO add debug variant
        pub static ref CMAKE_DEFS: std::collections::HashMap<&'static str, &'static str> = maplit::hashmap!{
            "BUILD_SHARED_LIBS" => "on",
            "LLAMA_SERVER_VERBOSE" => "off",
            "CMAKE_BUILD_TYPE" => "Release"
        };
    }

    pub fn build(
        src_dir: &str,
        build_dir: &str,
        cmake_defs: &std::collections::HashMap<&str, &str>,
        env: &std::collections::HashMap<&str, &str>,
        targets: &[&str],
    ) -> std::path::PathBuf {
        println!("cargo:warning=build with: cmake -S {src_dir} -B {build_dir} {cmake_defs:?}");
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
        for (k, v) in env.iter() {
            dd = dd.env(k, v);
        }
        for t in targets {
            dd = dd.build_target(t);
        }
        dd.build()
    }
}

mod linux {}

//#[cfg(feature = "build")]
#[cfg(target_os = "macos")]
mod macos {

    lazy_static::lazy_static! {
        static ref COMMON_DARWIN_DEFS: std::collections::HashMap<&'static str, &'static str> = maplit::hashmap!{
            "CMAKE_OSX_DEPLOYMENT_TARGET" => "11.3",
            "LLAMA_METAL_MACOSX_VERSION_MIN" => "11.3",
            "CMAKE_SYSTEM_NAME" => "Darwin",
            "LLAMA_METAL_EMBED_LIBRARY" => "on",
            "LLAMA_OPENMP" => "off"
        };

        static ref APPLE_IDENTITY: Option<String> = None;
    }

    fn sign(build_dir: &str) {
        if let Some(kk) = &*APPLE_IDENTITY {
            println!("cargo:warning=Signing {build_dir}/bin/*.dylib");
            for entry in glob::glob(&format!("{build_dir}/bin/*.dylib"))
                .expect("Failed to read glob pattern")
            {
                if let Ok(path) = entry {
                    let path = path.into_os_string().into_string().unwrap();
                    std::process::Command::new("codesign")
                        .arg("-f")
                        .arg("--timestamp")
                        .arg("--deep")
                        .arg("--options=runtime")
                        .arg(format!("--sign \"{}\"", kk))
                        .arg("--identifier nebula")
                        .arg(&path)
                        .status()
                        .expect("sign error");
                }
            }
        }
    }

    fn install(build_dir: &str, dist_dir: &str) {
        let pp = if let Ok(profile) = ::std::env::var("PROFILE") {
            profile
        } else {
            "Debug".to_string()
        };
        println!("cargo:warning=Installing binaries from {build_dir} to dist dir {dist_dir}");
        std::fs::create_dir_all(dist_dir).expect("can`t create dist directory");
        for entry in
            glob::glob(&format!("{build_dir}/build/*.dylib")).expect("Failed to read glob pattern")
        {
            if let Ok(path) = entry {
                let path = path.into_os_string().into_string().unwrap();
                println!("{path}");
                std::process::Command::new("cp")
                    .arg(path)
                    .arg(dist_dir)
                    .status()
                    .expect("sign error");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        lazy_static::lazy_static! {
            static ref ARCH: &'static str = "x86_64";
            static ref COMMON_CPU_DEFS: std::collections::HashMap<&'static str, &'static str> = super::COMMON_DARWIN_DEFS
                .iter()
                .chain(
                    maplit::hashmap!{
                        "CMAKE_SYSTEM_PROCESSOR" => "x86_64",
                        "CMAKE_OSX_ARCHITECTURES" => "x86_64",
                        "LLAMA_METAL" => "off",
                        "LLAMA_NATIVE" => "off"
                    }.iter())
                .map(|(k, v)| (*k, *v))
                .collect();
        }

        fn build_cpu(
            src_dir: &str,
            dist_dir: &str,
            _cmake_defs: &std::collections::HashMap<&str, &str>,
            targets: &[&str],
        ) {
            let cmake_defs: std::collections::HashMap<&str, &str> = COMMON_CPU_DEFS
                .iter()
                .chain(
                    maplit::hashmap! {
                        "LLAMA_ACCELERATE" => "off",
                        "LLAMA_BLAS" => "off",
                        "LLAMA_AVX" => "off",
                        "LLAMA_AVX2" => "off",
                        "LLAMA_AVX512" => "off",
                        "LLAMA_FMA" => "off",
                        "LLAMA_F16C" => "off"
                    }
                    .iter()
                    .chain(super::super::common::CMAKE_DEFS.iter()),
                )
                .map(|(k, v)| (*k, *v))
                .collect();
            println!("cargo:warning=Building LCD CPU");
            let build_dir = format!(
                "{}/darwin/{}/cpu",
                std::env::var("OUT_DIR").expect("No out dir found"),
                *ARCH
            );
            super::super::common::build(src_dir, &build_dir, &cmake_defs, targets);
            super::sign(&build_dir);
            super::install(&build_dir, &format!("{dist_dir}/cpu"));
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
                        "LLAMA_ACCELERATE" => "off",
                        "LLAMA_BLAS" => "off",
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
            println!("cargo:warning=Building AVX CPU");
            let build_dir = format!(
                "{}/darwin/{}/cpu_avx",
                std::env::var("OUT_DIR").expect("No out dir found"),
                *ARCH
            );
            super::super::common::build(src_dir, &build_dir, &cmake_defs, targets);
            super::sign(&build_dir);
            super::install(&build_dir, &format!("{dist_dir}/cpu_avx"));
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
                        "LLAMA_ACCELERATE" => "on",
                        "LLAMA_BLAS" => "off",
                        "LLAMA_AVX" => "on",
                        "LLAMA_AVX2" => "on",
                        "LLAMA_AVX512" => "off",
                        "LLAMA_FMA" => "on",
                        "LLAMA_F16C" => "on",
                        "-framework Accelerate" => "",
                        "-framework Foundation" => ""
                    }
                    .iter()
                    .chain(cmake_defs.iter()),
                )
                .map(|(k, v)| (*k, *v))
                .collect();
            println!("cargo:warning=Building AVX2 CPU");
            let build_dir = format!(
                "{}/darwin/{}/cpu_avx2",
                std::env::var("OUT_DIR").expect("No out dir found"),
                *ARCH
            );
            super::super::common::build(src_dir, &build_dir, &cmake_defs, targets);
            super::sign(&build_dir);
            super::install(&build_dir, &format!("{dist_dir}/cpu_avx2"));
        }

        pub fn bbuild() {
            build_cpu(
                *super::super::common::LLAMACPP_DIR,
                &format!("dist/darwin/{}/", *ARCH),
                &*super::super::common::CMAKE_DEFS,
                *super::super::common::CMAKE_TARGETS,
            );

            if ::std::is_x86_feature_detected!("avx") {
                build_cpu_avx(
                    *super::super::common::LLAMACPP_DIR,
                    &format!("dist/darwin/{}/", *ARCH),
                    &*super::super::common::CMAKE_DEFS,
                    *super::super::common::CMAKE_TARGETS,
                );
            }
            if ::std::is_x86_feature_detected!("avx2") {
                build_cpu_avx2(
                    *super::super::common::LLAMACPP_DIR,
                    &format!("dist/darwin/{}/", *ARCH),
                    &*super::super::common::CMAKE_DEFS,
                    *super::super::common::CMAKE_TARGETS,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod aarch64 {
        lazy_static::lazy_static! {
            static ref ARCH: &'static str = "arm64";
        }

        fn build_metal(
            src_dir: &str,
            dist_dir: &str,
            _cmake_defs: &std::collections::HashMap<&str, &str>,
            targets: &[&str],
        ) {
            let cmake_defs: std::collections::HashMap<&str, &str> = super::COMMON_DARWIN_DEFS
                .iter()
                .chain(
                    maplit::hashmap! {
                        "LLAMA_ACCELERATE" => "on",
                        "CMAKE_SYSTEM_PROCESSOR" => *ARCH,
                        "CMAKE_OSX_ARCHITECTURES" => *ARCH,
                        "LLAMA_METAL" => "on",
                    }
                    .iter()
                    .chain(super::super::common::CMAKE_DEFS.iter()),
                )
                .map(|(k, v)| (*k, *v))
                .collect();
            println!("cargo:warning=Building Metal");
            let build_dir = format!(
                "{}/darwin/{}/metal",
                std::env::var("OUT_DIR").expect("No out dir found"),
                *ARCH
            );
            super::super::common::build(
                src_dir,
                &build_dir,
                &cmake_defs,
                &maplit::hashmap! {
                    "EXTRA_LIBS" => "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
                },
                targets,
            );
            super::sign(&build_dir);
            super::install(&build_dir, &format!("{dist_dir}/cpu"));
        }

        pub fn bbuild() {
            build_metal(
                *super::super::common::LLAMACPP_DIR,
                &format!("dist/darwin/{}/", *ARCH),
                &*super::super::common::CMAKE_DEFS,
                *super::super::common::CMAKE_TARGETS,
            );
        }
    }

    pub fn bbuild() {
        #[cfg(target_arch = "x86_64")]
        x86::bbuild();
        #[cfg(target_arch = "aarch64")]
        aarch64::bbuild();
    }
}

//#[cfg(feature = "build")]
#[cfg(target_os = "windows")]
mod windows {

    lazy_static::lazy_static! {
        static ref LLAMACPP_DIR: &'static str = "llama.cpp";
        static ref CMAKE_TARGETS: &'static[&'static str] = &["llama", "llava"];
        //TODO add debug variant
        static ref CMAKE_DEFS: std::collections::HashMap<&'static str, &'static str> = maplit::hashmap!{
            "BUILD_SHARED_LIBS" => "on",
            "LLAMA_NATIVE" => "off",
            "LLAMA_OPENMP" => "off",
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
    ) -> std::path::PathBuf {
        println!("cargo:warning=build with: cmake -S {src_dir} -B {build_dir} {cmake_defs:?}");
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
        dd.build()
    }

    fn sign(build_dir: &str) {
        if let Some(kk) = &*KEY_CONTAINER {
            println!("cargo:warning=Signing {build_dir}/bin/*.exe  {build_dir}/bin/*.dll");
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
        let pp = if let Ok(profile) = ::std::env::var("PROFILE") {
            profile
        } else {
            "Debug".to_string()
        };
        println!("cargo:warning=Installing binaries to dist dir {dist_dir}");
        std::fs::create_dir_all(dist_dir).expect("can`t create dist directory");
        for entry in glob::glob(&format!("{build_dir}/build/bin/{pp}/*.dll"))
            .expect("Failed to read glob pattern")
        {
            if let Ok(path) = entry {
                let path = path.into_os_string().into_string().unwrap();
                powershell_script::run(&format!(
                    "copy-item -Path \"{path}\" -Destination \"{dist_dir}\" -Force"
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
        println!("cargo:warning=Building LCD CPU");
        let build_dir = format!(
            "{}/windows/{}/cpu",
            std::env::var("OUT_DIR").expect("No out dir found"),
            *ARCH
        );
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
        println!("cargo:warning=Building AVX CPU");
        let build_dir = format!(
            "{}/windows/{}/cpu_avx",
            std::env::var("OUT_DIR").expect("No out dir found"),
            *ARCH
        );
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
        println!("cargo:warning=Building AVX2 CPU");
        let build_dir = format!(
            "{}/windows/{}/cpu_avx2",
            std::env::var("OUT_DIR").expect("No out dir found"),
            *ARCH
        );
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
                "(get-item (\"{nvcc}\" | split-path | split-path)).Basename"
            ))
            .expect("can`t get cuda version")
            .stdout()
            .unwrap();
            let cuda_version = cuda_version.trim();
            let build_dir = format!(
                "{}/windows/{}/cuda_{cuda_version}",
                std::env::var("OUT_DIR").expect("No out dir found"),
                *ARCH
            );
            let disst_dir = format!("{dist_dir}/cuda_{cuda_version}");
            let cmake_defs: std::collections::HashMap<&str, &str> = CMAKE_DEFS
                .iter()
                .chain(
                    maplit::hashmap! {
                        "LLAMA_CUDA" => "ON",
                        "LLAMA_AVX" => "on",
                        "LLAMA_AVX2" => "off",
                        "CUDAToolkit_INCLUDE_DIR" => &cuda_include_dir,
                        "CMAKE_CUDA_FLAGS" => "-t8",
                        "CMAKE_CUDA_ARCHITECTURES" => &*CMAKE_CUDA_ARCHITECTURES
                    }
                    .iter(),
                )
                .map(|(k, v)| (*k, *v))
                .collect();
            println!("cargo:warning=Building CUDA GPU");
            build(src_dir, &build_dir, &cmake_defs, targets)
                .into_os_string()
                .into_string()
                .unwrap();
            sign(&build_dir);
            install(&build_dir, &disst_dir);
            println!("copying CUDA dependencies to {dist_dir}");
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
                    println!("{}", path.display());
                    let path = path.into_os_string().into_string().unwrap();
                    println!("{}", path);
                    powershell_script::run(&format!(
                        "copy-item -Path \"{path}\" -Destination \"{dist_dir}\" -Force"
                    ))
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

    //    #[cfg(feature = "build")]
    #[cfg(target_os = "windows")]
    windows::bbuild();

    //    #[cfg(feature = "build")]
    #[cfg(target_os = "macos")]
    macos::bbuild();
}
