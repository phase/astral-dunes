pub fn debug_memory(msg: impl AsRef<str>) {
    use systemstat::*;
    let sys = System::new();
    if let Ok(mem) = sys.memory() {
        println!("{}: {}/{}", msg.as_ref(), saturating_sub_bytes(mem.total, mem.free), mem.total);
    }
}
