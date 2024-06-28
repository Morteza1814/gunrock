#include <gunrock/algorithms/spmv.hxx>
#include <gunrock/algorithms/generate/random.hxx>
#include "spmv_cpu.hxx"

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 5) {
    std::cerr << "usage: ./bin/<program-name> -m filename.mtx -n num_runs" << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  // Filename to be read
  std::string filename = argument_array[2];
  uint32_t num_runs = std::stoi(argument_array[4]);
  // Load the matrix-market dataset into csr format.
  // See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  csr_t csr;
  csr.from_coo(coo);

  // --
  // Build graph

  // Convert the dataset you loaded into an `essentials` graph.
  // `memory_space_t::device` -> the graph will be created on the GPU.
  auto G = graph::build<memory_space_t::device>(properties, csr);

  std::vector<float> run_times;
  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> x(n_vertices);
  thrust::device_vector<weight_t> y(n_vertices);

  for (int i = 0; i < num_runs; i++) {
    gunrock::generate::random::uniform_distribution(x);

    // --
    // GPU Run
    run_times.push_back(gunrock::spmv::run(G, x.data().get(), y.data().get()));
  }

  // --
  // CPU Run

  thrust::host_vector<weight_t> y_h(n_vertices);
  float cpu_elapsed = spmv_cpu::run(csr, x, y_h);

  // --
  // Log + Validate
  int n_errors = util::compare(
      y.data().get(), y_h.data(), n_vertices,
      [=](const weight_t a, const weight_t b) {
        // TODO: needs better accuracy.
        return std::abs(a - b) > 1e-2;
      },
      true);

  gunrock::print::head(y, 40, "GPU y-vector");
  gunrock::print::head(y_h, 40, "CPU y-vector");

  // std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;

  float total_time = 0;
  for (int i = 0; i < num_runs; i++) {
    std::cout << "Run " << i << " Exec Time: " << run_times[i] << " (ms)" << std::endl;
    //get the average time
    total_time += run_times[i];
  }
  std::cout << "Average GPU Elapsed Time : " << (float)(total_time/num_runs) << " (ms)"
            << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}