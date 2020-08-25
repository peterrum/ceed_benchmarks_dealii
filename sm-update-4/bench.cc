#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_sm_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <chrono>
#include <vector>

#include "../common_code/renumber_dofs_for_mf.h"
#include "../sm-util/mpi.h"
#include "../sm-util/timers.h"

using namespace dealii;

using Number              = double;
using VectorizedArrayType = VectorizedArray<Number>;

const MPI_Comm comm = MPI_COMM_WORLD;

std::pair<unsigned int, std::vector<unsigned int>>
compute_grid(const unsigned int dim, unsigned int s)
{
  std::vector<unsigned int> subdivisions(dim);

  for (unsigned int i = 0; i < dim; ++i)
    subdivisions[i] = (i < (s % dim)) ? 2 : 1;

  return {/*refinements = */ s / dim, subdivisions};
}

template <int dim, int degree, int n_points_1d = degree + 1>
void
test(ConvergenceTable & table,
     const MPI_Comm     comm,
     const MPI_Comm     comm_sm,
     const unsigned int n_cyle)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // 1) create Triangulation
  parallel::distributed::Triangulation<dim> tria(comm);

  const auto pair = compute_grid(dim, n_cyle);

  const auto p1 = dim == 2 ? Point<dim>(0, 0) : Point<dim>(0, 0, 0);
  const auto p2 = dim == 2 ? Point<dim>(1, 1) : Point<dim>(1, 1, 1);

  GridGenerator::subdivided_hyper_rectangle(tria, pair.second, p1, p2);
  tria.refine_global(pair.first);

  // 2) create DoFHandler
  DoFHandler<dim> dof_handler(tria);
  FE_Q<dim>       fe(degree);
  dof_handler.distribute_dofs(fe);

  // 3) create MatrixFree
  AffineConstraints<Number> constraint;
  constraint.close();

  QGauss<1> quad(n_points_1d);

  MappingQGeneric<dim> mapping(1);

  typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;
  additional_data.overlap_communication_computation = true;

  if (true)
    {
      Renumber<dim, double> renum(0, 1, 2);
      renum.renumber(dof_handler, constraint, additional_data);
    }

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  // 4) run performance tests
  const auto run = [&](const auto &label, const auto &runnable) {
    const unsigned int n_repertitions = 1000;

    MPI_Barrier(comm);
    {
      for (unsigned int i = 0; i < 100; i++)
        runnable();
    }
    {
      MPI_Barrier(comm);
    }
    const auto temp = std::chrono::system_clock::now();
    MPI_Barrier(comm);
    {
      hyperdeal::ScopedLikwidTimerWrapper likwid(label);
      for (unsigned int i = 0; i < n_repertitions; i++)
        runnable();
    }
    MPI_Barrier(comm);

    const auto ms =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - temp)
        .count();

    const auto result =
      static_cast<double>(dof_handler.n_dofs()) * sizeof(Number) * n_repertitions / ms / 1000;

    table.add_value(label, result);
    table.set_scientific(label, true);

    return result;
  };

  const auto bps = [&](auto &vec1, auto &vec2, const std::string &label) {
    return std::pair<double, double>{
      run(label,
          [&]() {
            FEEvaluation<dim, degree, n_points_1d, 1, Number, VectorizedArrayType> phi(matrix_free);

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); cell++)
              {
                phi.reinit(cell);
                phi.gather_evaluate(vec1, true, false);

                for (unsigned int q = 0; q < phi.n_q_points; q++)
                  phi.submit_value(phi.get_value(q), q);

                phi.integrate_scatter(true, false, vec2);
              }
          }),
      run(label + "_comm", [&]() {
        FEEvaluation<dim, degree, n_points_1d, 1, Number, VectorizedArrayType> phi(matrix_free);
        matrix_free.template cell_loop<decltype(vec1), decltype(vec2)>(
          [&](const auto &, auto &dst, const auto &src, const auto cells) {
            for (unsigned int cell = cells.first; cell < cells.second; cell++)
              {
                phi.reinit(cell);
                phi.gather_evaluate(src, true, false);

                for (unsigned int q = 0; q < phi.n_q_points; q++)
                  phi.submit_value(phi.get_value(q), q);

                phi.integrate_scatter(true, false, dst);
              }
          },
          vec1,
          vec2);
      })};
  };

  // .. for LinearAlgebra::distributed::Vector
  const auto result_d = [&]() {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // create vectors
    VectorType vec1, vec2;
    matrix_free.initialize_dof_vector(vec1);
    matrix_free.initialize_dof_vector(vec2);

    table.add_value("size_local", Utilities::MPI::sum(vec1.get_partitioner()->local_size(), comm));
    table.add_value("size_ghost",
                    Utilities::MPI::sum(vec1.get_partitioner()->n_ghost_indices(), comm));

    // apply mass matrix
    return bps(vec1, vec2, "L:D:V");
  }();

  // .. for LinearAlgebra::SharedMPI::Vector
  const auto result_s = [&]() {
    using VectorType = LinearAlgebra::SharedMPI::Vector<Number>;

    // create vectors
    VectorType vec1, vec2;
    matrix_free.initialize_dof_vector(vec1, comm_sm);
    matrix_free.initialize_dof_vector(vec2, comm_sm);

    // apply mass matrix
    return bps(vec1, vec2, "L:S:V");
  }();

  table.add_value("speedup", result_s.first / result_d.first);
  table.set_scientific("speedup", true);
  table.add_value("speedup_comm", result_s.second / result_d.second);
  table.set_scientific("speedup_comm", true);
}

template <int dim>
void
test_dim(ConvergenceTable & table,
         const MPI_Comm     comm,
         const MPI_Comm     comm_sm,
         const unsigned int degree,
         const unsigned int n_refinements)
{
  switch (degree)
    {
      case 3:
        test<dim, 3>(table, comm, comm_sm, n_refinements);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

  ConditionalOStream pcout(std::cout, rank == 0);

  // print help page if requested
  if (argc == 1 || std::string(argv[1]) == "--help")
    {
      pcout << "mpirun -np 40 ./bench group_size dim degree n_refinements" << std::endl;
      return 0;
    }

  // read parameters form command line
  const unsigned int group_size    = argc < 2 ? 1 : atoi(argv[1]);
  const unsigned int dim           = argc < 3 ? 3 : atoi(argv[2]);
  const unsigned int degree        = argc < 4 ? 3 : atoi(argv[3]);
  const unsigned int n_refinements = argc < 5 ? 8 : atoi(argv[4]);

  hyperdeal::ScopedLikwidInitFinalize likwid;

  // create convergence table
  ConvergenceTable table;

  // run tests for different group sizes
  for (auto size : (group_size == 0 ? hyperdeal::mpi::create_possible_group_sizes(comm) :
                                      std::vector<unsigned int>{group_size}))
    {
      // create sm communicator
      MPI_Comm comm_sm;
      MPI_Comm_split(comm, rank / size, rank, &comm_sm);

      for (unsigned int n_cyles = dim; n_cyles < n_refinements; ++n_cyles)
        {
          table.add_value("group_size", size);
          // perform tests
          if (dim == 2)
            test_dim<2>(table, comm, comm_sm, degree, n_cyles);
          else if (dim == 3)
            test_dim<3>(table, comm, comm_sm, degree, n_cyles);
        }

      // free communicator
      MPI_Comm_free(&comm_sm);
    }

  // print convergence table
  if (pcout.is_active())
    table.write_text(pcout.get_stream());
}