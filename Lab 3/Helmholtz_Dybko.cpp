#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <algorithm>
#include <fstream>

int n = 2001;
double eps = 1e-8;

void matrix_print(const std::vector<double>& matrix, int m, int k) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			std::cout  << matrix[i * k + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

double f(double x, double y, double k)
{
	return 2.0 * std::sin(M_PI * y) + k * k * (1.0 - x) * x * std::sin(M_PI * y) + M_PI * M_PI * (1.0 - x) * x * std::sin(M_PI * y);
}

double sol(double x, double y)
{
	return (1.0 - x) * x * std::sin(M_PI * y);
}

void init_boundaries(std::vector<double>& local_grid, int local_rows, int start_row, int n) {
	for (int li = 0; li < local_rows; ++li) {
		local_grid[li * n + 0] = 0.0;      // x = 0
		local_grid[li * n + (n - 1)] = 0.0;  // x = 1
	}
}

double calculate_error(std::vector<double> &num_solution, double solution(double, double), double h){
	double error = 0.0;
//	double diff = 0.0;
//	#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			auto diff = std::fabs(num_solution[i * n + j] - solution(i*h, j*h));
			if (diff > error){
	//				#pragma omp critical
				error = std::max(diff, error);
			}
		}
	}
	return error;
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
//	MPI_Status stat[4];

	double max_diff_local = 0.0;
	double global_max_diff = 0.0;
	double t1 = 0.0, t2 = 0.0, t3 =0.0,  t_total = 0.0, t_wait = 0.0;

	const double* my_top_most = local_grid_old.data(); // первая строка моих данных в local_grid_old
	const double* my_bottom_most = local_grid_old.data() + (local_rows - 1) * n; // последняя строка моих данных в local_grid_old

	MPI_Send_init(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
	MPI_Recv_init(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);

	MPI_Send_init(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
	MPI_Recv_init(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);

	for (int iter = 0; iter < max_iter; ++iter) {
		max_diff_local = 0.0;

		if (rank == 0) {
			std::fill(top.begin(), top.end(), 0.0);
		}
		if (rank == size - 1) {
			std::fill(bottom.begin(), bottom.end(), 0.0);
		}

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
		if (send_type == 3)
		{
			// t1 = MPI_Wtime();

			// MPI_Isend(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
			// MPI_Irecv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);
			//
			// MPI_Isend(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
			// MPI_Irecv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);


			MPI_Startall(4, req);

			// t2 = MPI_Wtime();

			for (int li = 1; li < local_rows - 1; ++li) {
				int i = start_row + li;
				if (i < 1 or i > n - 2) continue;
				for (int j = 1; j < n - 1; ++j) {
					int index = li * n + j;
					double up = local_grid_old[index - n];
					double down  = local_grid_old[index + n];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid_old[index - 1] +
						local_grid_old[index + 1]) / denominator;
					double diff = std::fabs(local_grid[index] - local_grid_old[index]);
					max_diff_local = std::max(max_diff_local, diff);
				}

			}
			// t3 = MPI_Wtime();
			MPI_Waitall(4, req, MPI_STATUS_IGNORE);
			// t_total += (MPI_Wtime() - t1);
			// t_wait += (MPI_Wtime() - t3);

			int li=0; int i=start_row;
			if (i != 0 ) {
				for (int j = 1; j < n - 1; ++j) {
					int idx = j;
					double up = top[j];
					double down = local_grid_old[idx + n];
					local_grid[idx] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid_old[idx - 1] +
						local_grid_old[idx + 1]) /
						denominator;
					max_diff_local = std::max(max_diff_local, std::fabs(local_grid[idx] - local_grid_old[idx]));
				}
			}

			li=local_rows-1; i=start_row+li;
			if (i != n-1) {
				for (int j = 1; j < n - 1; ++j) {
					int idx = li * n + j;
					double up = local_grid_old[idx - n];
					double down = bottom[j];
					local_grid[idx] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid_old[idx - 1] +
						local_grid_old[idx + 1]) /
						denominator;
					max_diff_local = std::max(max_diff_local, std::fabs(local_grid[idx] - local_grid_old[idx]));
				}
			}

		}
		if (send_type == 1 or send_type == 2){
			for (int li = 0; li < local_rows; ++li) {
				int i = start_row + li;
				if (i < 1 or i > n - 2) continue;
				for (int j = 1; j < n - 1; ++j) {
					int index = li * n + j;
					double up = (li > 0) ? local_grid_old[index - n] : top[j];
					double down  = (li < local_rows - 1) ? local_grid_old[index + n] : bottom[j];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid_old[index - 1] +
						local_grid_old[index + 1]) / denominator;
					double diff = std::fabs(local_grid[index] - local_grid_old[index]);
					max_diff_local = std::max(max_diff_local, diff);
				}
			}
		}

		MPI_Allreduce(&max_diff_local, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (global_max_diff < eps) {
			if (rank == 0)
				std::cout << "Jacobi converged after " << (iter + 1) << std::endl;
			if (rank == 0 and send_type == 3) {
				printf("Non-blocking start: %.2e s, compute_internal: %.2e s, wait: %.2e s\n",
					t2-t1, t3-t2, t_wait);
			}
			return true;
		}
		local_grid.swap(local_grid_old);
	}

	if (rank == 0)
		std::cout << "Jacobi did NOT converge in " << max_iter << std::endl;
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
//	MPI_Status stat[4];

	double max_diff_local = 0.0;
	double global_max_diff = 0.0;
	const double* my_top_most = local_grid.data();
	const double* my_bottom_most = local_grid.data() + (local_rows - 1) * n;

	if (send_type == 3)
	{
		MPI_Send_init(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
		MPI_Recv_init(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);

		MPI_Send_init(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
		MPI_Recv_init(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);
	}

	for (int iter = 0; iter < max_iter; ++iter) {
		max_diff_local = 0.0;

		if (rank == 0) {
			std::fill(top.begin(), top.end(), 0.0);
		}
		if (rank == size - 1) {
			std::fill(bottom.begin(), bottom.end(), 0.0);
		}

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
		if (send_type == 3)
		{

			// MPI_Isend(my_bottom_most,n, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
			// MPI_Irecv(top.data(),n, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);
			//
			// MPI_Isend(my_top_most,n, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[2]);
			// MPI_Irecv(bottom.data(),n, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[3]);


			//
			MPI_Startall(4, req);

			//Red
			for (int li = 1; li < local_rows - 1; ++li) {
				int i = start_row + li;
				int j_start = (i % 2 == 0) ? 2 : 1;
				if (i < 1 or i > n - 2) continue;
				for (int j = j_start; j < n - 1; j += 2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up = local_grid[index - n];
					double down = local_grid[index + n];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						 up +
						 down +
						 local_grid[index + 1] +
						 local_grid[index - 1]) / denominator;
					auto diff = std::fabs(elem - local_grid[index]);
					if (diff > max_diff_local) {
						max_diff_local = std::max(diff, max_diff_local);
					}
				}
			}

			MPI_Wait(&req[0], MPI_STATUS_IGNORE);
			MPI_Wait(&req[1], MPI_STATUS_IGNORE);

			int li = 0; int i = start_row + li;
			if (i != 0 ) {
				int j_start = (i % 2 == 0) ? 2 : 1;
				for (int j = j_start; j < n - 1; j+=2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up = (rank == 0) ?  0.0 : top[j];
					double down = local_grid[index + n];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid[index + 1] +
						local_grid[index - 1]) /
						denominator;
					double diff = std::fabs(elem - local_grid[index]);
					max_diff_local = std::max(diff, max_diff_local);
				}
			}

			MPI_Wait(&req[2], MPI_STATUS_IGNORE);
			MPI_Wait(&req[3], MPI_STATUS_IGNORE);

			li = local_rows - 1; i = start_row + li;
			if (i != n-1) {
				int j_start = (i % 2 == 0) ? 2 : 1;
				for (int j = j_start; j < n - 1; j+=2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up =  local_grid[index - n];
					double down = (rank == size -1) ? 0.0 : bottom[j];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						up +
						down +
						local_grid[index + 1] +
						local_grid[index - 1]) / denominator;
					double diff = std::fabs(elem - local_grid[index]);
					max_diff_local = std::max(diff, max_diff_local);
				}
			}

			//Black
			for (int li = 0; li < local_rows; ++li) {
				int i = start_row + li;
				int j_start = (i % 2 == 0) ? 1 : 2;
				if (i < 1 or i > n - 2) continue;
				for (int j = j_start; j < n - 1; j += 2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up = (li > 0) ? local_grid[index - n] : top[j];
					double down = (li < local_rows - 1) ? local_grid[index + n] : bottom[j];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						 up +
						 down +
						 local_grid[index + 1] +
						 local_grid[index - 1]) / denominator;
					auto diff = std::fabs(elem - local_grid[index]);
					//	if (diff > max_diff_local) {
					max_diff_local = std::max(diff, max_diff_local);
					//	}
				}
			}
		}
		if (send_type == 1 or send_type == 2) {
			for (int li = 0; li < local_rows; ++li) {
				int i = start_row + li;
				int j_start = (i % 2 == 0) ? 2 : 1;
				if (i < 1 or i > n - 2) continue;
				for (int j = j_start; j < n - 1; j += 2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up = (li > 0) ? local_grid[index - n] : top[j];
					double down = (li < local_rows - 1) ? local_grid[index + n] : bottom[j];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						 up +
						 down +
						 local_grid[index + 1] +
						 local_grid[index - 1]) / denominator;
					auto diff = std::fabs(elem - local_grid[index]);
				//	if (diff > max_diff_local) {
						max_diff_local = std::max(diff, max_diff_local);
				//	}
				}
			}

			for (int li = 0; li < local_rows; ++li) {
				int i = start_row + li;
				int j_start = (i % 2 == 0) ? 1 : 2;
				if (i < 1 or i > n - 2) continue;
				for (int j = j_start; j < n - 1; j += 2) {
					int index = li * n + j;
					double elem = local_grid[index];
					double up = (li > 0) ? local_grid[index - n] : top[j];
					double down = (li < local_rows - 1) ? local_grid[index + n] : bottom[j];
					local_grid[index] =
						(h2 * f(h * i, h * j, k) +
						 up +
						 down +
						 local_grid[index + 1] +
						 local_grid[index - 1]) / denominator;
					auto diff = std::fabs(elem - local_grid[index]);
				//	if (diff > max_diff_local) {
						max_diff_local = std::max(diff, max_diff_local);
				//	}
				}
			}
		}

		MPI_Allreduce(&max_diff_local, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


		if (global_max_diff < eps) {
			if (rank == 0)
			{
				std::cout << "Red and black iterations converged after " << (iter + 1) << std::endl;
			}
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
	double h = 1.0 / (n-1);
	double k = 1.0/h;
//	double k = 10.0;
	if (rank == 0) {
		std::cout << "Number of nodes = " << size << std::endl;
		std::cout << "k = " << k << std::endl;
		std::cout << "Step = " << h << std::endl;
		std::cout << "Tolerance = " << eps << std::endl;
	}

	int base_rows = n / size;
	int leftover  = n % size;

	std::vector<int> counts(size), displs(size);
	displs[0] = 0;
	for (int i = 0; i < size; ++i) {
		int rows_i = (i < leftover) ? base_rows + 1 : base_rows;
		counts[i] = rows_i * n;
		if (i > 0) displs[i] = displs[i-1] + counts[i-1];
	}

	int local_rows = counts[rank] / n;
	int start_row  = displs[rank] / n;
	{
		if (rank == 0) {
			std::cout << "Jacoby Send + Recv" << std::endl;
		}
		std::vector<double> grid_jac_1(n * n, 0.0);
		std::vector<double> local_grid_jac_1(local_rows * n, 0.0);
		std::vector<double> local_grid_old_1 = local_grid_jac_1;
		init_boundaries(local_grid_jac_1, local_rows, start_row, n);
		init_boundaries(local_grid_old_1, local_rows, start_row, n);
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

		MPI_Gatherv(local_grid_jac_1.data(), local_rows * n, MPI_DOUBLE,
			grid_jac_1.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_jacoby += +omp_get_wtime();

		if (rank == 0 and jacoby_flag) {
			std::cout << "error = " << calculate_error(grid_jac_1, sol, h) << std::endl;
			std::cout << "time = " << time_jacoby << std::endl << std::endl;
			//		std::cout << "speedup = " << 24.9837/time_jacoby << std::endl << std::endl;
		}
	}
	{
		if (rank == 0) {
			std::cout << "Jacoby SendRecv" <<  std::endl;
		}

		std::vector<double> grid_jac_2(n * n, 0.0);
		std::vector<double> local_grid_jac_2(local_rows * n, 0.0);
		std::vector<double> local_grid_old_2 = local_grid_jac_2;
		init_boundaries(local_grid_jac_2, local_rows, start_row, n);
		init_boundaries(local_grid_old_2, local_rows, start_row, n);
		auto time_jacoby = -omp_get_wtime();
		auto jacoby_flag = Jacoby_MPI(local_grid_jac_2,
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
		MPI_Gatherv(local_grid_jac_2.data(), local_rows * n, MPI_DOUBLE,
			grid_jac_2.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_jacoby += +omp_get_wtime();

		if (rank == 0 and jacoby_flag) {
			std::cout << "error = " << calculate_error(grid_jac_2, sol, h) << std::endl;
			std::cout << "time = " << time_jacoby << std::endl << std::endl;
		}
	}
	{
		if (rank == 0) {
			std::cout << "Jacoby Isend + Irecv" << std::endl;
		}
		std::vector<double> grid_jac_3(n * n, 0.0);
		std::vector<double> local_grid_jac_3(local_rows * n, 0.0);
		std::vector<double> local_grid_old_3 = local_grid_jac_3;
		init_boundaries(local_grid_jac_3, local_rows, start_row, n);
		init_boundaries(local_grid_old_3, local_rows, start_row, n);
		auto time_jacoby = -omp_get_wtime();
		auto jacoby_flag = Jacoby_MPI(local_grid_jac_3,
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
		MPI_Gatherv(local_grid_jac_3.data(), local_rows * n, MPI_DOUBLE,
			grid_jac_3.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_jacoby += +omp_get_wtime();

		if (rank == 0 and jacoby_flag) {
			std::cout << "error = " << calculate_error(grid_jac_3, sol, h) << std::endl;
			std::cout << "time = " << time_jacoby << std::endl << std::endl;
			std::cout << "speedup = " << 265.056/time_jacoby << std::endl << std::endl;
		}
	}
	{
		if (rank == 0) {
			std::cout << "Red and black iterations Send + Recv" << std::endl;
		}
		std::vector<double> grid_rb_1(n * n, 0.0);

		std::vector<double> local_grid_rb_1(local_rows * n, 0.0);
		init_boundaries(local_grid_rb_1, local_rows, start_row, n);
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
		MPI_Gatherv(local_grid_rb_1.data(), local_rows * n, MPI_DOUBLE,
			grid_rb_1.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_rb += +omp_get_wtime();

		if (rank == 0 and rb_flag) {
			auto time_err = -omp_get_wtime();
			std::cout << "error = " << calculate_error(grid_rb_1, sol, h) << std::endl;
			time_err += omp_get_wtime();
			// std::cout << "time error= " << time_err << std::endl << std::endl;
			std::cout << "time = " << time_rb << std::endl << std::endl;
		}
	}
	{
		if (rank == 0) {
			std::cout << "Red and black iterations SendRecv" << std::endl;
		}
		std::vector<double> grid_rb_2(n * n, 0.0);
		std::vector<double> local_grid_rb_2(local_rows * n, 0.0);
		init_boundaries(local_grid_rb_2, local_rows, start_row, n);
		auto time_rb = -omp_get_wtime();
		auto rb_flag = Red_and_black_iterations_MPI(local_grid_rb_2,
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

		MPI_Gatherv(local_grid_rb_2.data(), local_rows * n, MPI_DOUBLE,
			grid_rb_2.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_rb += +omp_get_wtime();


		if (rank == 0 and rb_flag) {
			std::cout << "error = " << calculate_error(grid_rb_2, sol, h) << std::endl;
			std::cout << "time = " << time_rb << std::endl << std::endl;
		}
	}
	{
		if (rank == 0) {
			std::cout << "Red and black iterations Isend + Irecv" << std::endl;
		}
		std::vector<double> grid_rb_3(n * n, 0.0);
		std::vector<double> local_grid_rb_3(local_rows * n, 0.0);
		init_boundaries(local_grid_rb_3, local_rows, start_row, n);
		auto time_rb = -omp_get_wtime();
		auto rb_flag = Red_and_black_iterations_MPI(local_grid_rb_3,
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
		MPI_Gatherv(local_grid_rb_3.data(), local_rows * n, MPI_DOUBLE,
			grid_rb_3.data(), counts.data(), displs.data(), MPI_DOUBLE,
			0, MPI_COMM_WORLD);
		time_rb += +omp_get_wtime();


		if (rank == 0 and rb_flag) {
			std::cout << "error = " << calculate_error(grid_rb_3, sol, h) << std::endl;
			std::cout << "time = " << time_rb << std::endl << std::endl;
			std::cout << "speedup = " << 151.339/time_rb << std::endl << std::endl;
		}
	}
	MPI_Finalize();
	return 0;
}
