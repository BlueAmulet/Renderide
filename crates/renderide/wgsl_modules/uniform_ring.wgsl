#define_import_path uniform_ring

struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
