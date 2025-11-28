#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <mpi.h>

int n = 1000;
double eps = 1e-6;

double f(double x, double y, double k)
{
	return 2.0 * std::sin(M_PI * y) + k * k * (1.0 - x) * x * std::sin(M_PI * y) + M_PI * M_PI * (1.0 - x) * x * std::sin(M_PI * y);
}

double sol(double x, double y)
{
	return (1.0 - x) * x * std::sin(M_PI * y);
}

double calculate_error(std::vector<double> &num_solution, double solution(double, double), double h){
	double error = 0.0;
//	double diff = 0.0;
	#pragma omp parallel for collapse(2)
	for (int i = 1; i < n; ++i){
		for (int j = 1; j < n; ++j){
			auto diff = std::fabs(num_solution[i * n + j] - solution(i*h, j*h));
			if (diff > error){
				#pragma omp critical
				error = std::max(diff, error);
			}
		}
	}
	return error;
}

bool Jacoby(std::vector<double> &grid, double h, double k, int max_iter){
	double diff = 0.0;
	std::vector<double> grid_old(grid);
	double denominator = (4.0 + k * k * h * h);
	double h2 = h * h;
	double max_diff = 0.0;
	for (int iter = 0; iter < max_iter; ++iter ){
//		#pragma omp parallel for
		for (int i = 1; i < n-1; ++i)
			for (int j = 1; j < n-1; ++j){
				int index = i * n + j;
				grid[index] = (h2 * f(h * i, h * j, k)  +  grid_old[index + n] + grid_old[index -n]+ grid_old[index + 1] + grid_old[index - 1]) / denominator;
				diff = std::fabs(grid[index] - grid_old[index]);
				if (diff > max_diff) {
//					#pragma omp critical
					max_diff = std::max(diff, max_diff);
				}
			}
		if (max_diff < eps){
			std::cout << "Jacoby converged after " << iter << " iterations" << std::endl;
			return true;
		}
		grid.swap(grid_old);
		max_diff = 0.0;
 	}
	return false;
}

bool Jacoby_MPI(
	std::vector<double>& local_grid,
	std::vector<double>& local_grid_old,
	int local_rows,
	int start_row,
	int n,
	double h,
	double k,
	double eps,
	int max_iter,
	int rank,
	int size,
	int send_type)
{
	const double denominator = 4.0 + k * k * h * h;
	const double h2 = h * h;

	int upper = (rank == 0) ? MPI_PROC_NULL : rank - 1;
	int lower = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

	std::vector<double> top(n, 0.0);
	std::vector<double> bottom(n, 0.0);

	MPI_Request req[4];
	MPI_Status stat[4];

	double max_diff_local = 0.0;
	double global_max_diff = 0.0;

	for (int iter = 0; iter < max_iter; ++iter) {
		max_diff_local = 0.0;
		const double* my_top_most = local_grid_old.data(); // первая строка моих данных в local_grid_old
		const double* my_bottom_most = local_grid_old.data() + (local_rows - 1) * n; // последняя строка моих данных в local_grid_old

		if (send_type == 1){
			MPI_Send(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD);
			MPI_Recv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Send(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD);
			MPI_Recv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (send_type == 2)
		{
			MPI_Sendrecv(my_bottom_most, n, MPI_DOUBLE, lower, 0,
				         top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Sendrecv(my_top_most,n, MPI_DOUBLE, upper, 1,
				         bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		}
		else
		{
			MPI_Isend(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
			MPI_Irecv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);

			MPI_Isend(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
			MPI_Irecv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);

			MPI_Waitall(4, req, stat);
		}

		for (int li = 0; li < local_rows; ++li) {
			int i = start_row + li;
			if (i < 1 || i > n - 2) continue;
			//#pragma omp parallel for
			for (int j = 1; j < n - 1; ++j) {
				int index = li * n + j;
				double up = (li > 0) ? local_grid_old[index - n] : top[j];
				double down  = (li < local_rows - 1) ? local_grid_old[index + n] : bottom[j];
				local_grid[index] = (h2 * f(h * i, h * j, k) + up + down + local_grid_old[index - 1] + local_grid_old[index + 1]) / denominator;
				double diff = std::fabs(local_grid[index] - local_grid_old[index]);
				max_diff_local = std::max(max_diff_local, diff);
			}
		}
		MPI_Allreduce(&max_diff_local, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (global_max_diff < eps) {
			if (rank == 0)
				std::cout << "Jacobi converged after " << (iter + 1) << std::endl;
			return true;
		}
		local_grid.swap(local_grid_old);
	}

	if (rank == 0)
		std::cout << "Jacobi did NOT converge in " << max_iter << std::endl;
	return false;
}

//
//
bool Red_and_black_iterations(std::vector<double> &grid, double h, double k, int max_iter){
	double denominator = (4.0 + k * k * h * h);
	double h2 = h * h;
	double max_diff = 0.0;

	for (int iter = 0; iter < max_iter; ++iter ){
//		#pragma omp parallel for
		for (int i = 1; i < n - 1; ++i) {
			int j_start = (i % 2 == 0) ? 2 : 1;
			for (int j = j_start; j < n - 1; j += 2) {
				int index = i * n + j;
				double elem = grid[index];
				grid[index] =
					(h2 * f(h * i, h * j, k) +
					 grid[index + n] +
					 grid[index - n ] +
					 grid[index + 1] +
					 grid[index -1]) / denominator;
				auto diff = std::fabs(elem - grid[i * n + j]);
				if (diff > max_diff) {
//					#pragma omp critical
					max_diff = std::max(diff, max_diff);
				}
			}
		}

//		#pragma omp parallel for
		for (int i = 1; i < n - 1; ++i) {
			int j_start = (i % 2 == 0) ? 1 : 2;
			for (int j = j_start; j < n - 1; j += 2) {
				int index = i * n + j;
				double elem = grid[index];
				grid[index] =
					(h2 * f(h * i, h * j, k) +
					 grid[index + n] +
					 grid[index - n] +
					 grid[index + 1] +
					 grid[index - 1]) / denominator;
				auto diff = std::fabs(elem - grid[i * n + j]);
				if (diff > max_diff) {
//					#pragma omp critical
					{
					max_diff = std::max(diff, max_diff);
					}
				}
			}
		}
		if (max_diff < eps){
			std::cout << "Red and black iterations method converged after " << iter << " iterations" << std::endl;
			return true;
		}
		max_diff = 0.0;
	}
	return false;
}

bool Red_and_black_iterations_MPI(std::vector<double>& local_grid,
	int local_rows,
	int start_row,
	int n,
	double h,
	double k,
	double eps,
	int max_iter,
	int rank,
	int size,
	int send_type){
	double denominator = (4.0 + k * k * h * h);
	double h2 = h * h;
	double max_diff = 0.0;

	int upper = (rank == 0) ? MPI_PROC_NULL : rank - 1;
	int lower = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

	std::vector<double> top(n, 0.0);
	std::vector<double> bottom(n, 0.0);

	MPI_Request req[4];
	MPI_Status stat[4];

	double max_diff_local = 0.0;
	double global_max_diff = 0.0;

	for (int iter = 0; iter < max_iter; ++iter) {
		max_diff_local = 0.0;
		const double* my_top_most = local_grid.data(); // первая строка моих данных в local_grid_old
		const double* my_bottom_most = local_grid.data() + (local_rows - 1) * n; // последняя строка моих данных в local_grid_old

		if (send_type == 1){
			MPI_Send(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD);
			MPI_Recv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Send(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD);
			MPI_Recv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (send_type == 2)
		{
			MPI_Sendrecv(my_bottom_most, n, MPI_DOUBLE, lower, 0,
				top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Sendrecv(my_top_most,n, MPI_DOUBLE, upper, 1,
				bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		}
		else
		{
			MPI_Isend(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
			MPI_Irecv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);

			MPI_Isend(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
			MPI_Irecv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);

			MPI_Waitall(4, req, stat);
		}

		for (int li = 0; li < local_rows; ++li) {
			int i = start_row + li;
			int j_start = (i % 2 == 0) ? 2 : 1;
			if (i < 1 || i > n - 2) continue;
			for (int j = j_start; j < n - 1; j += 2) {
				int index = li * n + j;
				double elem = local_grid[index];
				double up = (li > 0) ? local_grid[index - n] : top[j];
				double down  = (li < local_rows - 1) ? local_grid[index + n] : bottom[j];
				local_grid[index] =
					(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid[index + 1] +
						local_grid[index -1]) / denominator;
				auto diff = std::fabs(elem - local_grid[index]);
				if (diff > max_diff_local) {
//					#pragma omp critical
					max_diff_local = std::max(diff, max_diff_local);
				}
			}
		}

		for (int li = 0; li < local_rows; ++li) {
			int i = start_row + li;
			int j_start = (i % 2 == 0) ? 1 : 2;
			if (i < 1 || i > n - 2) continue;
			for (int j = j_start; j < n - 1; j += 2) {
				int index = li * n + j;
				double elem = local_grid[index];
				double up = (li > 0) ? local_grid[index - n] : top[j];
				double down  = (li < local_rows - 1) ? local_grid[index + n] : bottom[j];
				local_grid[index] =
					(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid[index + 1] +
						local_grid[index - 1]) / denominator;
				auto diff = std::fabs(elem - local_grid[index]);
				if (diff > max_diff_local) {
//					#pragma omp critical
					{
						max_diff_local = std::max(diff, max_diff_local);
					}
				}
			}
		}
		MPI_Allreduce(&max_diff_local, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (global_max_diff < eps) {
			if (rank == 0)
				std::cout << "Red and black iterations converged after " << (iter + 1) << std::endl;
			return true;
		}
		max_diff = 0.0;
	}

	if (rank == 0)
		std::cout << "Red and black iterations did NOT converge in " << max_iter << std::endl;
	return false;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank;
	int size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double h = 1.0 / n;
	double k = 1.0 / h;
	if (rank == 0) {
		std::cout << "Number of nodes = " << size << std::endl;
		std::cout << "k = " << k << std::endl;
		std::cout << "Step = " << h << std::endl;
		std::cout << "Tolerance = " << eps << std::endl;
	}

	int local_rows = n / size;
	int start_row = rank * local_rows;

	if (rank == 0) {
		std::cout << "Jacoby Send + Recv" << std::endl;
	}
	std::vector<double> grid_jac_1(n * n, 0.0);
	std::vector<double> local_grid_jac_1(local_rows * n, 0.0);
	std::vector<double> local_grid_old_1 = local_grid_jac_1;
	auto time_jacoby = -omp_get_wtime();
	bool jacoby_flag = Jacoby_MPI(local_grid_jac_1,
		local_grid_old_1,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		1);
	MPI_Gather(local_grid_jac_1.data(), local_rows * n, MPI_DOUBLE,
		grid_jac_1.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_jacoby += +omp_get_wtime();

	if (rank == 0 and jacoby_flag) {
		std::cout << "Jacoby error = " << calculate_error(grid_jac_1, sol, h) << std::endl;
		std::cout << "Jacoby time = " << time_jacoby << std::endl << std::endl;
	}

	if (rank == 0) {
		std::cout << "Jacoby SendRecv" <<  std::endl;
	}

	std::vector<double> grid_jac_2(n * n, 0.0);
	std::vector<double> local_grid_jac_2(local_rows * n, 0.0);
	std::vector<double> local_grid_old_2 = local_grid_jac_2;
	time_jacoby = -omp_get_wtime();
	jacoby_flag = Jacoby_MPI(local_grid_jac_2,
		local_grid_old_2,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		2);
	MPI_Gather(local_grid_jac_2.data(), local_rows * n, MPI_DOUBLE,
		grid_jac_2.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_jacoby += +omp_get_wtime();

	if (rank == 0 and jacoby_flag) {
		std::cout << "Jacoby error = " << calculate_error(grid_jac_2, sol, h) << std::endl;
		std::cout << "Jacoby time = " << time_jacoby << std::endl << std::endl;
	}

	if (rank == 0) {
		std::cout << "Jacoby Isend + Irecv" << std::endl;
	}
	std::vector<double> grid_jac_3(n * n, 0.0);
	std::vector<double> local_grid_jac_3(local_rows * n, 0.0);
	std::vector<double> local_grid_old_3 = local_grid_jac_3;
	time_jacoby = -omp_get_wtime();
	jacoby_flag = Jacoby_MPI(local_grid_jac_3,
		local_grid_old_3,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		3);
	MPI_Gather(local_grid_jac_3.data(), local_rows * n, MPI_DOUBLE,
		grid_jac_3.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_jacoby += +omp_get_wtime();

	if (rank == 0 and jacoby_flag) {
		std::cout << "Jacoby error = " << calculate_error(grid_jac_3, sol, h) << std::endl;
		std::cout << "Jacoby time = " << time_jacoby << std::endl << std::endl;
	}

	if (rank == 0) {
		std::cout << "Red and black iterations Send + Recv" << std::endl;
	}
	std::vector<double> grid_rb_1(n * n, 0.0);

	std::vector<double> local_grid_rb_1(local_rows * n, 0.0);
	auto time_rb = -omp_get_wtime();
	bool rb_flag = Red_and_black_iterations_MPI(local_grid_rb_1,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		1);
	MPI_Gather(local_grid_rb_1.data(), local_rows * n, MPI_DOUBLE,
		grid_rb_1.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_rb += +omp_get_wtime();

	if (rank == 0 and rb_flag) {
		std::cout << "Red and black iterations error = " << calculate_error(grid_rb_1, sol, h) << std::endl;
		std::cout << "Red and black iterations time = " << time_rb << std::endl << std::endl;
	}

	if (rank == 0) {
		std::cout << "Red and black iterations SendRecv" << std::endl;
	}
	std::vector<double> grid_rb_2(n * n, 0.0);
	std::vector<double> local_grid_rb_2(local_rows * n, 0.0);
	time_rb = -omp_get_wtime();
	rb_flag = Red_and_black_iterations_MPI(local_grid_rb_2,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		2);
	MPI_Gather(local_grid_rb_2.data(), local_rows * n, MPI_DOUBLE,
		grid_rb_2.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_rb += +omp_get_wtime();

	if (rank == 0 and rb_flag) {
		std::cout << "Red and black iterations error = " << calculate_error(grid_rb_2, sol, h) << std::endl;
		std::cout << "Red and black iterations time = " << time_rb << std::endl << std::endl;
	}

	if (rank == 0) {
		std::cout << "Red and black iterations Isend + Irecv" << std::endl;
	}
	std::vector<double> grid_rb_3(n * n, 0.0);
	std::vector<double> local_grid_rb_3(local_rows * n, 0.0);
	time_rb = -omp_get_wtime();
	rb_flag = Red_and_black_iterations_MPI(local_grid_rb_3,
		local_rows,
		start_row,
		n,
		h,
		k,
		eps,
		10000,
		rank,
		size,
		3);
	MPI_Gather(local_grid_rb_3.data(), local_rows * n, MPI_DOUBLE,
		grid_rb_3.data(), local_rows * n, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	time_rb += +omp_get_wtime();

	if (rank == 0 and rb_flag) {
		std::cout << "Red and black iterations error = " << calculate_error(grid_rb_3, sol, h) << std::endl;
		std::cout << "Red and black iterations time = " << time_rb << std::endl << std::endl;
	}
	//	auto time_jacoby = -omp_get_wtime();
//	bool jacoby_flag = Jacoby(grid_jac_3, h, 1, 100000);
//	time_jacoby += +omp_get_wtime();
//	if (jacoby_flag){
//		std::cout << "Jacoby error = " << calculate_error(grid_jac_3, sol, h) << std::endl;
//		std::cout << "Jacoby time = " << time_jacoby << std::endl;
//	}
//	std::vector<double> grid_2(n*n, 0.0);
//	auto time_rb = -omp_get_wtime();
//	bool rb_flag = Red_and_black_iterations(grid_2, h, 1, 100000);
//	time_rb += omp_get_wtime();
//	if (rb_flag){
//		std::cout << "Red and black iterations method error = " << calculate_error(grid_2, sol, h) << std::endl;
//		std::cout << "Red and black iterations time = " << time_rb << std::endl;
//	}
//	std::cout << std::endl;
//
//	std::vector<double> grid_one(n*n, 0.0);
//
//	omp_set_num_threads(1);
//	auto time_jacoby_one = -omp_get_wtime();
//	bool jacoby_flag_one = Jacoby(grid_one, h, 1, 100000);
//	time_jacoby_one += +omp_get_wtime();
//	if (jacoby_flag_one){
//		std::cout << "Jacoby error = " << calculate_error(grid_one, sol, h) << std::endl;
//		std::cout << "Jacoby time = " << time_jacoby_one << std::endl;
//
//	}
//	std::vector<double> grid_2_one(n*n, 0.0);
//	auto time_rb_one = -omp_get_wtime();
//	bool rb_flag_one = Red_and_black_iterations(grid_2_one, h, 1, 100000);
//	time_rb_one += omp_get_wtime();
//	if (rb_flag_one){
//		std::cout << "Red and black iterations method error = " << calculate_error(grid_2_one, sol, h) << std::endl;
//		std::cout << "Red and black iterations time = " << time_rb_one << std::endl;
//	}
//
//	std::cout << std::endl;
//	std::cout << "Speedup Jacoby = " << time_jacoby_one/time_jacoby << std::endl;
//	std::cout << "Speedup Red and black iterations = " << time_rb_one/time_rb << std::endl;

	MPI_Finalize();
	return 0;
}
