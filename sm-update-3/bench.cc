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

template <int dim, int degree, int n_points_1d = degree + 1>
void
test(ConvergenceTable & /*table*/,
     const MPI_Comm     comm,
     const MPI_Comm     comm_sm,
     const unsigned int n_refinements)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // 1) create Triangulation
  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

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

  LinearAlgebra::SharedMPI::Vector<Number> vec_sm;
  matrix_free.initialize_dof_vector(vec_sm, comm_sm);


  const auto &sm_view = matrix_free.get_vector_partitioner(comm_sm)->get_sm_view();

  auto &dof_info = const_cast<internal::MatrixFreeFunctions::DoFInfo &>(matrix_free.get_dof_info());

  for (auto &i : dof_info.dof_indices)
    i = sm_view[i];

  using VectorType = LinearAlgebra::SharedMPI::Vector<Number>;

  // create vectors
  VectorType vec1, vec2;
  matrix_free.initialize_dof_vector(vec1, comm_sm);
  matrix_free.initialize_dof_vector(vec2, comm_sm);


  for (unsigned int i = 0;
       i < matrix_free.get_vector_partitioner()->locally_owned_range().n_elements();
       i++)
    vec2.begin()[i] =
      matrix_free.get_vector_partitioner()->locally_owned_range().nth_index_in_set(i);

  FEEvaluation<dim, degree, n_points_1d, 1, Number, VectorizedArrayType> phi(matrix_free);
  matrix_free.template cell_loop<VectorType, VectorType>(
    [&](const auto &, auto &, const auto &src, const auto cells) {
      for (unsigned int cell = cells.first; cell < cells.second; cell++)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);

          for (unsigned int i = 0; i < phi.dofs_per_cell; i++)
            {
              for (auto v : phi.begin_dof_values()[i])
                printf("%3d", static_cast<int>(v));
              std::cout << std::endl;
            }
        }
    },
    vec1,
    vec2);

  std::cout << std::endl << std::endl;
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
      case 1:
        test<dim, 1>(table, comm, comm_sm, n_refinements);
        break;
      case 2:
        test<dim, 2>(table, comm, comm_sm, n_refinements);
        break;
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
  if (std::string(argv[1]) == "--help")
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
      // table.add_value("group_size", size);

      // create sm communicator
      MPI_Comm comm_sm;
      MPI_Comm_split(comm, rank / size, rank, &comm_sm);

      // perform tests
      if (dim == 2)
        test_dim<2>(table, comm, comm_sm, degree, n_refinements);
      else if (dim == 3)
        test_dim<3>(table, comm, comm_sm, degree, n_refinements);

      // free communicator
      MPI_Comm_free(&comm_sm);
    }

  // print convergence table
  // if (pcout.is_active())
  //  table.write_text(pcout.get_stream());
}