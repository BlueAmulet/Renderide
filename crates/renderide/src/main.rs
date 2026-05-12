//! Renderer binary entry point.

fn main() {
    match renderide::run() {
        Ok(exit) => std::process::exit(exit.process_code()),
        Err(e) => {
            if logger::is_initialized() {
                logger::error!("{e}");
            } else {
                #[expect(clippy::print_stderr, reason = "logger failed to initialize")]
                {
                    eprintln!("renderide: {e}");
                }
            }
            std::process::exit(1);
        }
    }
}
