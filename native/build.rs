use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn source_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("native source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            source_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let mut files = Vec::new();
    source_files(&root.join("src"), &mut files);
    files.push(root.join("build.rs"));
    files.push(root.join("assets/structured-token-recipe-v1.json"));
    files.push(root.join("assets/structured-token-recipe-public-exact-v2.json"));
    files.push(root.join("assets/card-instance-token-recipe-corrected-v2.json"));
    files.push(root.join("assets/card-instance-validation-pins-v2.json"));
    files.push(root.join("assets/card-instance-defense-v3-release.json"));
    files.push(root.join("assets/card-instance-formal-ppo-v1.json"));
    files.push(root.join("assets/card-instance-formal-ppo-group-pack-v1.json"));
    files.push(root.join("assets/card-instance-tree3-layout-recipe-v1.json"));
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        hash.update(
            file.strip_prefix(&root)
                .expect("source relative path")
                .as_os_str()
                .as_encoded_bytes(),
        );
        hash.update([0]);
        hash.update(fs::read(file).expect("source bytes"));
        hash.update([0]);
    }
    println!(
        "cargo:rustc-env=GFB_NATIVE_SOURCE_SHA256={:x}",
        hash.finalize()
    );
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    println!(
        "cargo:rustc-env=GFB_NATIVE_CARGO_LOCK_SHA256={:x}",
        Sha256::digest(fs::read(lock).expect("Cargo.lock bytes"))
    );
    let rustc = Command::new(env::var_os("RUSTC").expect("RUSTC"))
        .arg("--version")
        .output()
        .expect("rustc version");
    assert!(rustc.status.success(), "rustc version command failed");
    println!(
        "cargo:rustc-env=GFB_NATIVE_RUSTC={}",
        String::from_utf8(rustc.stdout)
            .expect("rustc version UTF-8")
            .trim()
    );
    for (key, output) in [
        ("TARGET", "TARGET"),
        ("PROFILE", "PROFILE"),
        ("CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS"),
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        let value = env::var(key).unwrap_or_default().replace('\u{1f}', " ");
        println!("cargo:rustc-env=GFB_NATIVE_{output}={value}");
    }
}
