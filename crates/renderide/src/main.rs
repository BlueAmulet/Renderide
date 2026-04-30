//! Renderer binary entry point.

fn main() {
    match renderide::run() {
        Ok(exit) => std::process::exit(exit.process_code()),
        Err(e) => {
            logger::error!("{e}");
            std::process::exit(1);
        }
    }
}
