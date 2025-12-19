#include <iostream>
#include <vector>
#include <iomanip>
#include <limits>
#include <random>
#include <omp.h>

const int n = 4*1024;
const int b = 64;

void fill(
	std::vector<double>& A)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		{
			A[i * n + j] = std::cos(i + j);
		}
}

void fill_random(std::vector<double>& A) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0, 10);
	std::vector<double> v(n);
	for (auto& x : A) {
		x = dist(gen);
	}
}

void zero(std::vector<double>& A)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			A[i * n + j] = 0.0;
}

void matrix_print(const std::vector<double>& matrix, int m, int k) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			std::cout << std::setprecision(4) << matrix[i * k + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}


double residual_LU(const std::vector<double>& A, const std::vector<double>& LU) {
	double normA = 0.0;
	double normDiff = 0.0;
	int n = static_cast<int>(std::sqrt(A.size()));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double sum = 0.0;

			for (int k = 0; k <= std::min(i, j); ++k) {
				double L_ik = (i == k) ? 1.0 : LU[i * n + k];
				double U_kj = LU[k * n + j];
				double term = L_ik * U_kj;

				if (!std::isfinite(term)) {
					std::cerr << "[Error] NaN/Inf encountered during multiplication at (i=" << i
							  << ", j=" << j << ", k=" << k << ").\n";
					return std::numeric_limits<double>::quiet_NaN();
				}

				sum += term;
			}

			double diff = A[i * n + j] - sum;
			if (!std::isfinite(diff)) {
				std::cerr << "[Error] NaN/Inf encountered in diff at (i=" << i
						  << ", j=" << j << ").\n";
				return std::numeric_limits<double>::quiet_NaN();
			}

			normA += A[i * n + j] * A[i * n + j];
			normDiff += diff * diff;
		}
	}
	double residual = std::sqrt(normDiff / normA);
	if (!std::isfinite(residual)) {
		std::cerr << "Residual value is NaN or Inf.\n";
		return std::numeric_limits<double>::quiet_NaN();
	}

	return residual;
}


void LU(std::vector<double>& A, int size) {
//	int size = std::sqrt(A.size());
	for (int i = 0; i < size - 1; ++i) {
		double diag = A[i * size + i];
		for (int j = i + 1; j < size; ++j) {
			A[j * size + i] /= diag;
			double a_ji = A[j * size + i];
			double* row_j = &A[j * size];
			const double* row_i = &A[i * size];
			for (int k = i + 1; k < size; ++k)
				row_j[k] -= a_ji * row_i[k];
		}
	}
}

void LU_local(std::vector<double>& A, int i0, int b) {
	for (int i = i0; i < i0 + b; ++i) {
		double diag = A[i * n + i];
		for (int j = i + 1; j < i0 + b; ++j) {
			A[j * n + i] /= diag;
			double a_ji = A[j * n + i];
			double* row_j = &A[j * n];
			const double* row_i = &A[i * n];
			for (int k = i + 1; k < i0 + b; ++k)
				row_j[k] -= a_ji * row_i[k];
		}
	}
}
// ( A22 A23 )
// ( A32 A33 )

//void LU_block_Demmel(std::vector<double>& A) {
//	auto time_LU_block = 0.0;
//	auto time_LU_loop1 = 0.0;
//	auto time_LU_loop2 = 0.0;
//	auto time_LU_loop3 = 0.0;
//
//	for (int bl = 0; bl < n; bl += b) {
//
//		time_LU_block -= omp_get_wtime();
//		// Разложение блока A(bl:bl+b-1, bl:bl+b-1)
//		LU_local(A, bl, b);
//		time_LU_block += omp_get_wtime();
//
//		time_LU_loop1 -= omp_get_wtime();
//		// Вычисление U23 = L22^{-1} * A23
////		for (int ii = bl; ii < bl + b; ++ii) {
////			for (int jj = bl + b; jj < n; ++jj) {
////				for (int k = bl; k < ii; ++k)
////					A[ii * n + jj] -= A[ii * n + k] * A[k * n + jj];
////			}
////		}
//		for (int k = bl; k < bl + b; ++k) {
//			double *rowk = &A[k * n]; // указатель на начало строки k
//			for (int i = k + 1; i < bl + b; ++i) {
//				double *rowii = &A[i * n]; // указатель на начало строки i
//				double aik = rowii[k]; // A[i, k]
//				for (int j = bl + b; j < n; j += b) {
//					int jj1 = std::min(n, j + b);
//					for (int jj = j; jj < jj1; ++jj) {
//						rowii[jj] -= aik * rowk[jj];
//					}
//				}
//			}
//		}
//		time_LU_loop1 += omp_get_wtime();
//
//		time_LU_loop2 -= omp_get_wtime();
//		// L32 = A32 * U22^{-1}
////		for (int ii = bl + b; ii < n; ++ii) {
////			for (int jj = bl; jj < bl + b; ++jj) {
////				for (int k = bl; k < jj; ++k)
////					A[ii * n + jj] -= A[ii * n + k] * A[k * n + jj];
////				A[ii * n + jj] /= A[jj * n + jj];
////			}
////		}
//		std::vector<double> col_buf(b);
//
//		for (int j = bl; j < bl + b; ++j) {
//			for (int k = 0; k < j - bl; ++k) {
//				col_buf[k] = A[(bl + k) * n + j];
//			}
//			double divisor = A[j * n + j];
//			for (int i = bl + b; i < n; ++i) {
//				double* Aii = &A[i * n];
//				double sum = Aii[j];
//				for (int k = 0; k < j - bl; ++k) {
//					sum -= Aii[bl + k] * col_buf[k];
//				}
//				Aii[j] = sum / divisor;
//			}
//		}
//		time_LU_loop2 += omp_get_wtime();
//
////		time_LU_loop3 -= omp_get_wtime();
////		 A33 := A33 - L32 * U23
////		for (int i = bl + b; i < n; ++i) { // по строкам A33
////			for (int j = bl + b; j < n; ++j) { // по столбцам A33
////				double sum = 0.0;
////				for (int k = bl; k < bl + b; ++k)
////					sum += A[i * n + k] * A[k * n + j];
////				A[i * n + j] -= sum;
////			}
////		}
//		time_LU_loop3 -= omp_get_wtime();
//
//		int pb = std::min(b, n - bl); // реальный размер блока
//		std::vector<double> buf(b * b); // буфер для U23
//
//		for (int j0 = bl + b; j0 < n; j0 += b) {
//			int jmax = std::min(n, j0 + b);
//			int jlen = jmax - j0;
//
//			for (int jj = 0; jj < jlen; ++jj) { // копирование U23(bl : bl + pb, j0 : j0 + jlen)
//				double* buf_col = &buf[jj * pb];  // начало "столбца" в буфере
//				double* src = &A[bl * n + j0 + jj];
//				for (int kk = 0; kk < pb; ++kk) {
//					buf_col[kk] = *src;
//					src += n;
//				}
//			}
//
//			for (int i0 = bl + b; i0 < n; i0 += b) {
//				int imax = std::min(n, i0 + b);
//				for (int i = i0; i < imax; ++i) { // цикл строкам блока
//					double* row_i = &A[i * n + j0];
//					double* l_row = &A[i * n + bl];
//					for (int jj = 0; jj < jlen; ++jj) {
//						double sum = 0.0;
//						double* buf_col = &buf[jj * pb];
//						for (int kk = 0; kk < pb; ++kk){
//							sum += l_row[kk] * buf_col[kk]; // l_rok[kk] = A[i, bl + kk]
//						}
//						row_i[jj] -= sum; // row_i[jj] = A[i, j0 + jj].
//					}
//				}
//			}
//		}
//		time_LU_loop3 += omp_get_wtime();
//	}
//	std::cout << "Time for LU in block = " << time_LU_block << std::endl;
//	std::cout << "Time for loop1 in block = " << time_LU_loop1 << std::endl;
//	std::cout << "Time for loop2 in block = " << time_LU_loop2 << std::endl;
//	std::cout << "Time for loop3 in block = " << time_LU_loop3 << std::endl;
//}

void LU_block_Demmel_new(std::vector<double>& A) {
//	auto time_LU_block = 0.0;
////	auto time_LU_loop3_new = 0.0;
//	auto time_LU_loop1 = 0.0;
//	auto time_LU_loop2 = 0.0;
//	auto time_LU_loop3 = 0.0;

	std::vector<double> A22(b * b);
	std::vector<double> A32((n - b) * b);
	std::vector<double> A23((n - b) * b);
	#pragma omp parallel shared(b, A, A22, A32, A23) default(none) if (omp_get_max_threads() > 1)
	{
	for (int bl = 0; bl < n; bl += b) {
		int size = std::min(b, n - bl);
		int d = n - bl - size;

		// Разложение блока A(bl:bl+b-1, bl:bl+b-1)
//		time_LU_block -= omp_get_wtime();
		// Заполнение A23
		#pragma omp single nowait
		{
			for (int i = 0; i < size; ++i) {
				const double* src = &A[(bl + i) * n + bl + size];
				double* dst = &A23[i * d];
				for (int j = 0; j < d; ++j)
					dst[j] = src[j];
			}
		}
		// Заполнение A32
#pragma omp single nowait
		{
			for (int i = 0; i < d; ++i) {
				const double* src = &A[(bl + size + i) * n + bl];
				double* dst = &A32[i * size];
#pragma omp simd
				for (int j = 0; j < size; ++j)
					dst[j] = src[j];
			}
		}

		#pragma omp single
		{
//			LU_local(A, bl, b);

			// Заполнение A22
			for (int i = 0; i < size; ++i) {
				const double* src = &A[(bl + i) * n + bl];
				double* dst = &A22[i * size];
				#pragma omp simd
				for (int j = 0; j < size; ++j)
					dst[j] = src[j];
			}
//			matrix_print(A22, b, b);
			LU(A22, b);
//			matrix_print(A22, b, b);
		}

//		time_LU_block += omp_get_wtime();
//		time_LU_loop1 -= omp_get_wtime();
//		int d = n - bl - b;


		// Вычисление U23 = L22^{-1} * A23
		// U23[i, j] = A23[i,j] - Sum_k L22[i,k]*U23[k,j]
		#pragma omp single
		{
			for (int i = 0; i < size; ++i) {
				double* row_i = &A23[i * d];
				for (int k = 0; k < i; ++k) {
					double Lik = A22[i * size + k]; // L22[i,k]
					const double* row_k = &A23[k * d];
					#pragma omp simd
					for (int j = 0; j < d; ++j)
						row_i[j] -= Lik * row_k[j];
				}
			}
		}

//		time_LU_loop1 += omp_get_wtime();

//		time_LU_loop2 -= omp_get_wtime();


		// L32 = A32 * U22^{-1}
		// L32[i,j] = (A32[i,j] - Sum_k L32[i,k]*U22[k,j]) / U22[j,j]
		#pragma omp for
		for (int i = 0; i < d; ++i) {
			double* rowA32 = &A32[i * size];
			for (int j = 0; j < size; ++j) {
				double sum = rowA32[j];
				#pragma omp simd reduction(+:sum)
				for (int k = 0; k < j; ++k) {
					sum += - rowA32[k] * A22[k * size + j]; // U22[k,j]
				}
				rowA32[j] = sum / A22[j * size + j];
			}
		}
//		time_LU_loop2 += omp_get_wtime();


//		time_LU_loop3 -= omp_get_wtime();
		// A33 = A33 - L32 * U23
		// A33[i,j] -= Sum_k L32[i,k]*U23[k, j]
		#pragma omp for
		for (int i = 0; i < d; ++i) {
			double* rowA33 = &A[(bl + size + i) * n + (bl + size)]; // начало i-й строки в A33
			for (int k = 0; k < size; ++k) {
				double lval = A32[i * size + k]; // L32[i,k]
				const double* rowU23 = &A23[k * d]; // U23 строка k
				#pragma omp simd
				for (int j = 0; j < d; ++j) {
					rowA33[j] -= lval * rowU23[j];
				}
			}
		}

//		time_LU_loop3 += omp_get_wtime();

		// Копирование блока A22 обратно в A
		#pragma omp single nowait
		{
			for (int i = 0; i < size; ++i) {
				double* dst = &A[(bl + i) * n + bl];   // указатель на начало строки (bl + i) в A
				const double* src = &A22[i * size]; // указатель на начало строки i в A22
				#pragma omp simd
				for (int j = 0; j < size; ++j)
					dst[j] = src[j];
			}
		}

		// Копирование блока A23 обратно в A
		#pragma omp single nowait
		{
			for (int i = 0; i < size; ++i) {
				double* dst = &A[(bl + i) * n + bl + size]; // начало подматрицы A23 в A
				const double* src = &A23[i * d];         // начало строки i в A23
				#pragma omp simd
				for (int j = 0; j < d; ++j)
					dst[j] = src[j];
			}
		}

		// Копирование блока A32 обратно в A
		#pragma omp single nowait
		{
			for (int i = 0; i < d; ++i) {
				double* dst = &A[(bl + i + size) * n + bl]; // начало подматрицы A32 в A
				const double* src = &A32[i * size];         // начало строки i в A32
				#pragma omp simd
				for (int j = 0; j < size; ++j)
					dst[j] = src[j];
			}
		}
		#pragma omp barrier
	}
};

//	std::cout << "Time for LU in block = " << time_LU_block << std::endl;
//	std::cout << "Time for loop1 in block = " << time_LU_loop1 << std::endl;
//	std::cout << "Time for loop2 in block = " << time_LU_loop2 << std::endl;
//	std::cout << "Time for loop3 in block = " << time_LU_loop3 << std::endl;
}

//void LU_block_Demmel_parallel(std::vector<double>& A) {
//	auto time_LU_block = 0.0;
//	auto time_LU_loop1 = 0.0;
//	auto time_LU_loop2 = 0.0;
//	auto time_LU_loop3 = 0.0;
//
//	for (int bl = 0; bl < n; bl += b) {
//
//		time_LU_block -= omp_get_wtime();
//		// Разложение блока A(bl:bl+b-1, bl:bl+b-1)
//		LU_local(A, bl, b);
//		time_LU_block += omp_get_wtime();
//
//		time_LU_loop1 -= omp_get_wtime();
//		// Вычисление U23 = L22^{-1} * A23
//		for (int i = bl; i < bl + b; ++i) {
//			double *rowii = &A[i * n]; // указатель на начало строки i
//			for (int k = bl; k < i; ++k) {
//				double *rowk = &A[k * n]; // указатель на начало строки k
//				double aik = rowii[k]; // A[i, k]
//				for (int j = bl + b; j < n; j += b) {
//					int jj1 = std::min(n, j + b);
//					#pragma omp simd
//					for (int jj = j; jj < jj1; ++jj) {
//						rowii[jj] -= aik * rowk[jj];
//					}
//				}
//			}
//		}
//		time_LU_loop1 += omp_get_wtime();
//
//		time_LU_loop2 -= omp_get_wtime();
//		// L32 = A32 * U22^{-1}
////		std::vector<double> col_buf(b);
//
//		std::vector<double> col_buf(b);
//
//		for (int j = bl; j < bl + b; ++j) {
//			for (int k = 0; k < j - bl; ++k) {
//				col_buf[k] = A[(bl + k) * n + j];
//			}
//			double divisor = A[j * n + j];
//			#pragma omp parallel for shared(bl, divisor, col_buf, A, j) default(none) if (omp_get_max_threads() > 1)
//			for (int i = bl + b; i < n; ++i) {
//				double* Aii = &A[i * n];
//				double sum = Aii[j];
//				#pragma omp simd
//				for (int k = 0; k < j - bl; ++k) {
//					sum -= Aii[bl + k] * col_buf[k];
//				}
//				Aii[j] = sum / divisor;
//			}
//		}
//		time_LU_loop2 += omp_get_wtime();
//
//		time_LU_loop3 -= omp_get_wtime();
//		// A33 := A33 - L32 * U23
//		int pb = std::min(b, n - bl); // реальный размер блока
////		std::vector<double> buf(b * b); // буфер для U23
//		#pragma omp parallel for shared(bl, n, pb, A) default(none) if (omp_get_max_threads() > 1)
//		for (int j0 = bl + b; j0 < n; j0 += b) {
//			std::vector<double> buf(b * b); // буфер для U23
//			int jmax = std::min(n, j0 + b);
//			int jlen = jmax - j0;
//			for (int jj = 0; jj < jlen; ++jj) { // копирование U23(bl : bl + pb, j0 : j0 + jlen)
//				double* buf_col = &buf[jj * pb];  // начало "столбца" в буфере
//				double* src = &A[bl * n + j0 + jj];
//				for (int kk = 0; kk < pb; ++kk) {
//					buf_col[kk] = *src;
//					src += n;
//				}
//			}
//
//			//#pragma omp parallel for schedule(static)
//			for (int i0 = bl + b; i0 < n; i0 += b) {
//				int imax = std::min(n, i0 + b);
//				for (int i = i0; i < imax; ++i) { // цикл по строкам блока
//					double* row_i = &A[i * n + j0];
//					double* l_row = &A[i * n + bl];
//					for (int jj = 0; jj < jlen; ++jj) {
//						double sum = 0.0;
//						double* buf_col = &buf[jj * pb];
//						#pragma omp simd
//						for (int kk = 0; kk < pb; ++kk) {
//							sum += l_row[kk] * buf_col[kk]; // l_row[kk] = A[i, bl + kk]
//						}
//						row_i[jj] -= sum; // row_i[jj] = A[i, j0 + jj].
//					}
//				}
//			}
//		}
//		time_LU_loop3 += omp_get_wtime();
//	}
//	std::cout << "Time for LU in parallel = " << time_LU_block << std::endl;
//	std::cout << "Time for loop1 in parallel = " << time_LU_loop1 << std::endl;
//	std::cout << "Time for loop2 in parallel = " << time_LU_loop2 << std::endl;
//	std::cout << "Time for loop3 in parallel = " << time_LU_loop3 << std::endl;
//}

int main(int agrc, char** argv){
	std::cout << "Matrix size (n) = " << n << std::endl;
	std::cout << "Block size (b) = " << b << std::endl;
	std::cout << "Number of threads = " << omp_get_num_procs() << std::endl;
	std::cout << '\n';
	std::vector<double> A(n*n);
	fill_random(A);


//	auto A1(A);
//	auto time_LU = -omp_get_wtime();
//	LU(A1);
//	time_LU += omp_get_wtime();
//	std::cout << "Middle Element " << A1[n/2*n + n/2] << std::endl;
//	std::cout << "Time for LU = " << time_LU << std::endl;
//	std::cout << '\n';
//	std::cout << "Residual LU "<< residual_LU(A, A1) << std::endl;
//	auto A2(A);
//	auto time_LU_block = -omp_get_wtime();
//	LU_block_Demmel_new(A2);
//	time_LU_block += omp_get_wtime();
//	std::cout << "Middle Element " << A2[n/2*n + n/2] << std::endl;
//	std::cout << "Time for Demmel LU = " << time_LU_block << std::endl;
//	std::cout << "Residual parallel LU " << residual_LU(A, A2) << std::endl;
////	std::cout << "speedup (classic/block) = " << time_LU/time_LU_block << std::endl;
//	std::cout << '\n';

//	omp_set_num_threads(1);
//	auto A22(A);
//	auto time_LU_block_parallel_one = -omp_get_wtime();
//	LU_block_Demmel_new(A22);
//	time_LU_block_parallel_one += omp_get_wtime();
//	std::cout << "Middle Element " << A22[n/2*n + n/2] << std::endl;
//	std::cout << "Time for LU parallel one thread = " << time_LU_block_parallel_one << std::endl;
//	std::cout << '\n';
//	std::cout << "Residual block LU " << residual_LU(A, A22) << std::endl;

//	for (int i = 2; i <= 18; i++){
////		num_threads = i;
//		omp_set_num_threads(i);
//		std::cout << "Number of threads: " << i << std::endl;


	auto A3(A);
	auto time_LU_block_parallel = -omp_get_wtime();
	LU_block_Demmel_new(A3);
	time_LU_block_parallel += omp_get_wtime();
	std::cout << "Middle Element " << A3[n/2*n + n/2] << std::endl;
	std::cout << "Time for LU parallel = " << time_LU_block_parallel << std::endl;
//	std::cout << "Residual parallel LU " << residual_LU(A, A3) << std::endl;


//		std::cout << "speedup (block_one_thread/block_parallel) = " << time_LU_block_parallel_one/time_LU_block_parallel << std::endl;
//		std::cout << "efficiency (speedup/num_threads) = " << time_LU_block_parallel_one/time_LU_block_parallel/omp_get_num_threads() << std::endl << std::endl;;
//	std::cout << "Residual parallel LU " << residual_LU(A, A3) << std::endl;
//	}
//	omp_set_num_threads(num_threads);
//	auto A3(A);
//	auto time_LU_block_parallel = -omp_get_wtime();
//	LU_block_Demmel_parallel(A3);
//	time_LU_block_parallel += omp_get_wtime();
//	std::cout << "Middle Element " << A3[n/2*n + n/2] << std::endl;
//	std::cout << "Time for LU parallel = " << time_LU_block_parallel << std::endl;
//	std::cout << "speedup (block/block_parallel) = " << time_LU_block/time_LU_block_parallel << std::endl;
//	std::cout << "speedup (block_one_thread/block_parallel) = " << time_LU_block_parallel_one/time_LU_block_parallel << std::endl;
//	std::cout << "efficiency (speedup/num_threads) = " << time_LU_block/time_LU_block_parallel/num_threads << std::endl;
//	std::cout << "Residual parallel LU " << residual_LU(A, A3) << std::endl;
}
