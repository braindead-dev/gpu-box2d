// gb_pybind.cpp. The pybind11 module that exposes the batched gpu-box2d driver to
// Python. It wraps GBBatch (bindings/gb_batch.cuh) so an RL or simulation layer drives
// N independent Box2D worlds and reads the state out as numpy arrays. The API is
// game-agnostic: it speaks bodies, joints, the static boundary, and the step.
//
// State is returned as numpy arrays shaped by world and body:
//   positions   [NW, max_bodies, 2]   (x, y)
//   angles      [NW, max_bodies]
//   velocities  [NW, max_bodies, 3]   (vx, vy, angular)
//   awake       [NW, max_bodies]      (uint8)
//   body_count  [NW]                  (int32, includes the ground slot)
// Slot 0 is the static ground body. Slots past a world's body_count read as 0.
//
// Build (host, CPU): see bindings/setup.py. The module steps host-side and is
// bit-identical to a single-threaded host Box2D 2.3.0. A CUDA build of the same driver
// steps the same seeded state on the device through the SoA-global production path.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gb_batch.cuh"

namespace py = pybind11;

// A Python-facing batch of worlds. Owns a GBBatch and presents the seeding and the
// read-back as methods that return numpy arrays.
class PyBatch {
public:
    explicit PyBatch(int n_worlds) : b_(n_worlds) {}

    int n_worlds() const { return b_.NW; }
    int max_bodies() const { return GB_MAX_BODIES; }
    int num_ground_edges() const { return GB_N_EDGES; }

    void set_ground_edge(int world, int edge, float ax, float ay, float bx, float by){
        check_world(world); check_edge(edge);
        gbBatchSetGroundEdge(b_, world, edge, ax, ay, bx, by);
    }

    int add_circle(int world, float px, float py, float radius, float inv_mass, float inv_i, int body_type){
        check_world(world);
        return gbBatchAddCircle(b_, world, px, py, radius, inv_mass, inv_i, body_type);
    }

#ifdef GB_ENABLE_POLYGONS
    int add_box(int world, float px, float py, float hx, float hy, float inv_mass, float inv_i, int body_type){
        check_world(world);
        return gbBatchAddBox(b_, world, px, py, hx, hy, inv_mass, inv_i, body_type);
    }
#endif

    void set_velocity(int world, int body, float vx, float vy, float w_ang){
        check_world(world); check_body(world, body);
        gbBatchSetVelocity(b_, world, body, vx, vy, w_ang);
    }

    void set_angle(int world, int body, float angle){
        check_world(world); check_body(world, body);
        gbBatchSetAngle(b_, world, body, angle);
    }

#ifdef GB_ENABLE_JOINTS
    int add_revolute_joint(int world, int body_a, int body_b,
                           float anchor_ax, float anchor_ay, float anchor_bx, float anchor_by){
        check_world(world); check_body(world, body_a); check_body(world, body_b);
        return gbBatchAddRevoluteJoint(b_, world, body_a, body_b, anchor_ax, anchor_ay, anchor_bx, anchor_by);
    }
#endif

    void step(int substeps){
        if (substeps < 1) throw std::invalid_argument("substeps must be >= 1");
        gbBatchStep(b_, substeps);
    }

    py::array_t<float> positions() const {
        py::array_t<float> out({b_.NW, GB_MAX_BODIES, 2});
        gbBatchGetPositions(b_, static_cast<float*>(out.request().ptr));
        return out;
    }
    py::array_t<float> angles() const {
        py::array_t<float> out({b_.NW, GB_MAX_BODIES});
        gbBatchGetAngles(b_, static_cast<float*>(out.request().ptr));
        return out;
    }
    py::array_t<float> velocities() const {
        py::array_t<float> out({b_.NW, GB_MAX_BODIES, 3});
        gbBatchGetVelocities(b_, static_cast<float*>(out.request().ptr));
        return out;
    }
    py::array_t<uint8_t> awake() const {
        py::array_t<uint8_t> out({b_.NW, GB_MAX_BODIES});
        gbBatchGetAwake(b_, static_cast<unsigned char*>(out.request().ptr));
        return out;
    }
    py::array_t<int32_t> body_count() const {
        py::array_t<int32_t> out(b_.NW);
        gbBatchGetBodyCount(b_, static_cast<int*>(out.request().ptr));
        return out;
    }

private:
    GBBatch b_;
    void check_world(int w) const {
        if (w < 0 || w >= b_.NW) throw std::out_of_range("world index out of range");
    }
    void check_edge(int e) const {
        if (e < 0 || e >= GB_N_EDGES) throw std::out_of_range("ground edge index out of range");
    }
    void check_body(int w, int body) const {
        if (body < 0 || body >= b_.worlds[w].bodyCount) throw std::out_of_range("body index out of range");
    }
};

PYBIND11_MODULE(gpu_box2d, m){
    m.doc() = "gpu-box2d: batched, bit-faithful Box2D 2.3.0 worlds. Drive N independent "
              "physics worlds and read per-world body state as numpy arrays.";

    // body-type constants, matching b2BodyType.
    m.attr("STATIC_BODY") = GB_STATIC_BODY;
    m.attr("DYNAMIC_BODY") = GB_DYNAMIC_BODY;

    py::class_<PyBatch>(m, "Batch",
        "A batch of N independent Box2D worlds. Seed bodies, the static boundary, and "
        "joints, then step every world and read the state out as numpy arrays. Slot 0 "
        "of each world is the static ground body.")
        .def(py::init<int>(), py::arg("n_worlds"),
             "Create a batch of n_worlds empty worlds, each with a static ground body.")
        .def_property_readonly("n_worlds", &PyBatch::n_worlds,
             "Number of worlds in the batch.")
        .def_property_readonly("max_bodies", &PyBatch::max_bodies,
             "Maximum bodies per world (the per-world capacity, including the ground).")
        .def_property_readonly("num_ground_edges", &PyBatch::num_ground_edges,
             "Number of static ground-edge slots per world.")
        .def("set_ground_edge", &PyBatch::set_ground_edge,
             py::arg("world"), py::arg("edge"), py::arg("ax"), py::arg("ay"), py::arg("bx"), py::arg("by"),
             "Set a static ground edge segment from (ax, ay) to (bx, by) for one world.")
        .def("add_circle", &PyBatch::add_circle,
             py::arg("world"), py::arg("px"), py::arg("py"), py::arg("radius"),
             py::arg("inv_mass"), py::arg("inv_i"), py::arg("body_type"),
             "Add a circle body and return its slot. body_type is STATIC_BODY or "
             "DYNAMIC_BODY. inv_mass and inv_i are the inverse mass and inertia.")
#ifdef GB_ENABLE_POLYGONS
        .def("add_box", &PyBatch::add_box,
             py::arg("world"), py::arg("px"), py::arg("py"), py::arg("hx"), py::arg("hy"),
             py::arg("inv_mass"), py::arg("inv_i"), py::arg("body_type"),
             "Add an axis-aligned box body with half-extents (hx, hy) and return its slot.")
#endif
        .def("set_velocity", &PyBatch::set_velocity,
             py::arg("world"), py::arg("body"), py::arg("vx"), py::arg("vy"), py::arg("w"),
             "Set a body's linear velocity (vx, vy) and angular velocity w.")
        .def("set_angle", &PyBatch::set_angle,
             py::arg("world"), py::arg("body"), py::arg("angle"),
             "Set a body's orientation angle in radians.")
#ifdef GB_ENABLE_JOINTS
        .def("add_revolute_joint", &PyBatch::add_revolute_joint,
             py::arg("world"), py::arg("body_a"), py::arg("body_b"),
             py::arg("anchor_ax"), py::arg("anchor_ay"), py::arg("anchor_bx"), py::arg("anchor_by"),
             "Pin body_a and body_b at the given body-local anchors with a revolute joint.")
#endif
        .def("step", &PyBatch::step, py::arg("substeps") = 1,
             "Step every world substeps times.")
        .def("positions", &PyBatch::positions,
             "Body positions, shape [n_worlds, max_bodies, 2] (x, y).")
        .def("angles", &PyBatch::angles,
             "Body angles, shape [n_worlds, max_bodies].")
        .def("velocities", &PyBatch::velocities,
             "Body velocities, shape [n_worlds, max_bodies, 3] (vx, vy, angular).")
        .def("awake", &PyBatch::awake,
             "Body awake flags, shape [n_worlds, max_bodies], uint8.")
        .def("body_count", &PyBatch::body_count,
             "Live body count per world, shape [n_worlds], int32.");
}
