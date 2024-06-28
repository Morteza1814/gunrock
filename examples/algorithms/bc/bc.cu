#include <gunrock/algorithms/bc.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;


// --
// Define types

using vertex_t = int;
using edge_t = int;
using weight_t = float;

struct Edge {
  vertex_t u;
  vertex_t v;
  weight_t w;

  Edge(vertex_t u, vertex_t v, weight_t w = 1.0) : u(u), v(v), w(w) {
  }
};

using EdgeList = std::vector<Edge>;
using VertexDegree = std::pair<vertex_t, int>;  // Pair of vertex id and degree

EdgeList ReadInMTX(std::ifstream &in, bool &needs_weights) {
  EdgeList el;
  std::string start, object, format, field, symmetry, line;
  in >> start >> object >> format >> field >> symmetry >> std::ws;

  if (start != "%%MatrixMarket") {
    std::cout << ".mtx file did not start with %%MatrixMarket" << std::endl;
    std::exit(-21);
  }
  if ((object != "matrix") || (format != "coordinate")) {
    std::cout << "only allow matrix coordinate format for .mtx" << std::endl;
    std::exit(-22);
  }
  if (field == "complex") {
    std::cout << "do not support complex weights for .mtx" << std::endl;
    std::exit(-23);
  }
  bool read_weights;
  if (field == "pattern") {
    read_weights = false;
  } else if ((field == "real") || (field == "double") || (field == "integer")) {
    read_weights = true;
  } else {
    std::cout << "unrecognized field type for .mtx" << std::endl;
    std::exit(-24);
  }
  bool undirected;
  if (symmetry == "symmetric") {
    undirected = true;
  } else if ((symmetry == "general") || (symmetry == "skew-symmetric")) {
    undirected = false;
  } else {
    std::cout << "unsupported symmetry type for .mtx" << std::endl;
    std::exit(-25);
  }

  // Skip all comment lines
  while (std::getline(in, line)) {
    if (line[0] != '%') {
      break;
    }
  }

  // Read the dimensions and non-zeros line explicitly
  std::istringstream dimensions_stream(line);
  int64_t m = 0, n = 0, nonzeros = 0;
  if (!(dimensions_stream >> m >> n >> nonzeros)) {
    std::cout << "Error parsing matrix dimensions and non-zeros" << std::endl;
    std::exit(-28);
  }

  if (m != n) {
    std::cout << m << " " << n << " " << nonzeros << std::endl;
    std::cout << "matrix must be square for .mtx" << std::endl;
    std::exit(-26);
  }

  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream edge_stream(line);
    vertex_t u, v;
    weight_t w = 1.0;
    edge_stream >> u >> v;
    if (read_weights) {
      edge_stream >> w;
    }
    u -= 1;
    v -= 1;
    el.push_back(Edge(u, v, w));
    if (undirected) {
      el.push_back(Edge(v, u, w));
    }
  }
  needs_weights = !read_weights;
  return el;
}

std::vector<VertexDegree> ReadFileAndReturnSortedDegrees(const std::string &filename) {
  bool needs_weights;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Couldn't open file " << filename << std::endl;
    std::exit(-2);
  }

  EdgeList edges = ReadInMTX(file, needs_weights);
  file.close();

  // Compute degrees
  std::unordered_map<vertex_t, int> vertex_degrees;
  for (const auto &edge : edges) {
    vertex_degrees[edge.u]++;
  }

  // Convert to vector and sort
  std::vector<VertexDegree> vertex_degree_pairs(vertex_degrees.begin(), vertex_degrees.end());
  std::sort(vertex_degree_pairs.begin(), vertex_degree_pairs.end(),
            [](const VertexDegree &a, const VertexDegree &b) {
    return a.second > b.second; // Sort in descending order
  });

  return vertex_degree_pairs;
}

using csr_t =
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

void parse_source_string_top(const std::string& source_str, std::vector<int>* source_vect, int n_vertices, int n_runs, const std::string& filename) {
  if (source_str == "") {
    // Read and sort vertices by degree
    std::vector<VertexDegree> sorted_degrees = ReadFileAndReturnSortedDegrees(filename);

    // Select top n_runs vertices
    for (int i = 0; i < n_runs && i < sorted_degrees.size(); i++) {
      std::cout << sorted_degrees[i].first << " " << sorted_degrees[i].second << std::endl;
      source_vect->push_back(sorted_degrees[i].first);
    }
  } else {
    std::stringstream ss(source_str);
    while (ss.good()) {
      std::string source;
      getline(ss, source, ',');
      int source_int;
      try {
        source_int = std::stoi(source);
      } catch (...) {
        std::cout << "Error: Invalid source"
                  << "\n";
        exit(1);
      }
      if (source_int >= 0 && source_int < n_vertices) {
        source_vect->push_back(source_int);
      } else {
        std::cout << "Error: Invalid source"
                  << "\n";
        exit(1);
      }
    }
    if (source_vect->size() == 1) {
      source_vect->insert(source_vect->end(), n_runs - 1, source_vect->at(0));
    }
  }
}

void test_bc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Betweenness Centrality");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(coo);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> bc_values(n_vertices);

  // Parse sources
  std::vector<int> source_vect;
  // gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
  //                                       n_vertices, params.num_runs);

  parse_source_string_top(params.source_string, &source_vect,
                                        n_vertices, params.num_runs, params.filename);

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(
        gunrock::bc::run(G, source_vect[i], bc_values.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "bc",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // --
  // Log

  std::cout << "Single source : " << source_vect.back() << "\n";
  print::head(bc_values, 40, "GPU bc values");
  // std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
  //           << " (ms)" << std::endl;
  float total_time = 0;
  for (int i = 0; i < params.num_runs; i++) {
    std::cout << "Run " << i << " Exec Time: " << run_times[i] << " (ms)" << std::endl;
    //get the average time
    total_time += run_times[i];
  }
  std::cout << "Average GPU Elapsed Time : " << (float)(total_time/params.num_runs) << " (ms)"
            << std::endl;
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
