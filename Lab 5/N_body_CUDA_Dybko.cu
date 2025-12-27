#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>


//#define G 6.67e-11
//#define eps2 1e-20
#define block_size 32

using T = float;
constexpr T G = 6.67*1e-11;
constexpr T eps2 = 1e-20;

struct Bodies{
	T* m;
	T* rx, *ry, *rz;
	T* vx, *vy, *vz;
	T* ax, *ay, *az;
	T n;
};

struct RKBuffers {
	Bodies k1, k2, k3, k4;
};

//#define CUDA_CHECK(call) \
//do { \
//	cudaError_t err = call; \
//	if (err != cudaSuccess) { \
//		std:: cerr << "CUDA error: " << cudaGetErrorString(err) \
//		<< " at " << __FILE__ << ":" << __LINE__ << std::endl; \
//	exit(EXIT_FAILURE); \
//	} \
//} while(0)

void read_data(const std::string& filename,
    int& n,
    std::vector<T>& m,
    std::vector<T>& rx, std::vector<T>& ry, std::vector<T>& rz,
    std::vector<T>& vx, std::vector<T>& vy, std::vector<T>& vz)
{
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Не удалось открыть файл " + filename);
	}

	file >> n;

	m.resize(n);
	rx.resize(n); ry.resize(n); rz.resize(n);
	vx.resize(n); vy.resize(n); vz.resize(n);

	for (int i = 0; i < n; ++i) {
		file >> m[i]
			 >> rx[i] >> ry[i] >> rz[i]
			 >> vx[i] >> vy[i] >> vz[i];
	}
}


void initialize_output_files(int n, std::vector<std::ofstream>& files)
{
	files.resize(n);
	for (int i = 0; i < n; ++i) {
		std::string filename = "body_" + std::to_string(i+1) + ".txt";
		files[i].open(filename, std::ios::out | std::ios::trunc);

		if (!files[i].is_open()) {
			throw std::runtime_error("Не удалось открыть файл " + filename);
		}
		files[i] << std::fixed << std::setprecision(16);
	}
}


void write_positions(T t, const Bodies& bodies, std::vector<std::ofstream>& files)
{
	int n = bodies.n;

	std::vector<T> rx(n), ry(n), rz(n);

	cudaMemcpy(rx.data(), bodies.rx, n * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(ry.data(), bodies.ry, n * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(rz.data(), bodies.rz, n * sizeof(T), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) {
		files[i] << t << " "
				 << rx[i] << " "
				 << ry[i] << " "
				 << rz[i] << "\n";
	}
}


void generate_random_bodies(
	int n,
	std::vector<T>& m,
	std::vector<T>& rx, std::vector<T>& ry, std::vector<T>& rz,
	std::vector<T>& vx, std::vector<T>& vy, std::vector<T>& vz,
	T min_mass = 1e26, T max_mass = 1e30,
	T pos_min  = -1e10, T pos_max  = 1e10,
	T vel_min  = -1e5,  T vel_max  = 1e5)
{
	m.resize(n);
	rx.resize(n); ry.resize(n); rz.resize(n);
	vx.resize(n); vy.resize(n); vz.resize(n);

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<T> mass_dist(min_mass, max_mass);
	std::uniform_real_distribution<T> pos_dist(pos_min, pos_max);
	std::uniform_real_distribution<T> vel_dist(vel_min, vel_max);

	for (int i = 0; i < n; ++i) {
		m[i]  = mass_dist(gen);

		rx[i] = pos_dist(gen);
		ry[i] = pos_dist(gen);
		rz[i] = pos_dist(gen);

		vx[i] = vel_dist(gen);
		vy[i] = vel_dist(gen);
		vz[i] = vel_dist(gen);
	}
}


void allocate_bodies_device(Bodies& bodies, int n) {
	bodies.n = n;
	cudaMalloc(&bodies.m, n * sizeof(T));
	cudaMalloc(&bodies.rx, n * sizeof(T));
	cudaMalloc(&bodies.ry, n * sizeof(T));
	cudaMalloc(&bodies.rz, n * sizeof(T));
	cudaMalloc(&bodies.vx, n * sizeof(T));
	cudaMalloc(&bodies.vy, n * sizeof(T));
	cudaMalloc(&bodies.vz, n * sizeof(T));
	cudaMalloc(&bodies.ax, n * sizeof(T));
	cudaMalloc(&bodies.ay, n * sizeof(T));
	cudaMalloc(&bodies.az, n * sizeof(T));
}

void copy_to_device(Bodies& bodies,
	const std::vector<T>& m,
	const std::vector<T>& rx, const std::vector<T>& ry, const std::vector<T>& rz,
	const std::vector<T>& vx, const std::vector<T>& vy, const std::vector<T>& vz)
{
	int size = bodies.n * sizeof(T);
	cudaMemcpy(bodies.m, m.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.rx, rx.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.ry, ry.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.rz, rz.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.vx, vx.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.vy, vy.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(bodies.vz, vz.data(), size, cudaMemcpyHostToDevice);
}

void free_bodies_device(Bodies& bodies) {
	cudaFree(bodies.m);
	cudaFree(bodies.rx);
    cudaFree(bodies.ry);
    cudaFree(bodies.rz);
	cudaFree(bodies.vx);
    cudaFree(bodies.vy);
    cudaFree(bodies.vz);
	cudaFree(bodies.ax);
    cudaFree(bodies.ay);
    cudaFree(bodies.az);
}

__global__ void compute_acceleration_kernel(
    const T*  m,
    const T*  rx, const T* ry, const T* rz,
    T* ax, T* ay, T* az,
    int n)
{
    __shared__ T sh_m[block_size];
    __shared__ T sh_x[block_size];
    __shared__ T sh_y[block_size];
    __shared__ T sh_z[block_size];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;

    T xi = 0, yi = 0, zi = 0;
    if (i < n) {
        xi = rx[i]; yi = ry[i]; zi = rz[i];
    }

    T a_x = 0.0, a_y = 0.0, a_z = 0.0;
	int batch_size = (n + block_size - 1) / block_size;
    for (int batch = 0; batch < batch_size; ++batch) {
        int j = batch * block_size + idx;
        if (j < n) {
            sh_m[idx] = m[j];
            sh_x[idx] = rx[j];
            sh_y[idx] = ry[j];
            sh_z[idx] = rz[j];
        } else {
            sh_m[idx] = 0;
            sh_x[idx] = 0;
            sh_y[idx] = 0;
            sh_z[idx] = 0;
        }

        __syncthreads();

        if (i < n) {
            for (int k = 0; k < block_size; ++k) {
                T dx = sh_x[k] - xi;
                T dy = sh_y[k] - yi;
                T dz = sh_z[k] - zi;

            	T delta = dx*dx + dy*dy + dz*dz + eps2;
            	T multiplier = G * sh_m[k] / (sqrt(delta) * delta);

                a_x += multiplier * dx;
                a_y += multiplier * dy;
                a_z += multiplier * dz;
            }
        }

        __syncthreads();
    };

    if (i < n) {
        ax[i] = a_x;
        ay[i] = a_y;
        az[i] = a_z;
    }
}

//__global__ void save_state_kernel(
//	const T* vx, const T* vy, const T* vz,
//	const T* ax, const T* ay, const T* az,
//	T* vx_out, T* vy_out, T* vz_out,
//	T* ax_out, T* ay_out, T*  az_out,
//	int n)
//{
//	int i = blockIdx. x * blockDim.x + threadIdx.x;
//	if (i >= n) return;
//
//	vx_out[i] = vx[i];
//	vy_out[i] = vy[i];
//	vz_out[i] = vz[i];
//	ax_out[i] = ax[i];
//	ay_out[i] = ay[i];
//	az_out[i] = az[i];
//}

__global__ void runge_kutta_4_k(
    const T* rx, const T* ry, const T* rz,
    const T* vx_b, const T* vy_b, const T* vz_b,
    const T* vx, const T* vy, const T* vz,
    const T* ax, const T* ay, const T* az,
    T* rx_out, T* ry_out, T* rz_out,
    T* vx_out, T* vy_out, T* vz_out,
    T tau, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    rx_out[i] = rx[i] + tau * vx[i];
    ry_out[i] = ry[i] + tau * vy[i];
    rz_out[i] = rz[i] + tau * vz[i];

    vx_out[i] = vx_b[i] + tau * ax[i];
    vy_out[i] = vy_b[i] + tau * ay[i];
    vz_out[i] = vz_b[i] + tau * az[i];
}


__global__ void runge_kutta_4_final_step(
    T* rx, T* ry, T* rz,
    T* vx, T* vy, T* vz,
    const T* v1x, const T* v1y, const T* v1z,  // k1 v
    const T* v2x, const T* v2y, const T* v2z,  // k2 v
    const T* v3x, const T* v3y, const T* v3z,  // k3 v
    const T* v4x, const T* v4y, const T* v4z,  // k4 v
    const T* a1x, const T* a1y, const T* a1z,  // k1 a
    const T* a2x, const T* a2y, const T* a2z,  // k2 a
    const T* a3x, const T* a3y, const T* a3z,  // k3 a
    const T* a4x, const T* a4y, const T* a4z,  // k4 a
    T tau_6, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    rx[i] += tau_6 * (v1x[i] + 2.0*v2x[i] + 2.0*v3x[i] + v4x[i]);
    ry[i] += tau_6 * (v1y[i] + 2.0*v2y[i] + 2.0*v3y[i] + v4y[i]);
    rz[i] += tau_6 * (v1z[i] + 2.0*v2z[i] + 2.0*v3z[i] + v4z[i]);

    vx[i] += tau_6 * (a1x[i] + 2.0*a2x[i] + 2.0*a3x[i] + a4x[i]);
    vy[i] += tau_6 * (a1y[i] + 2.0*a2y[i] + 2.0*a3y[i] + a4y[i]);
    vz[i] += tau_6 * (a1z[i] + 2.0*a2z[i] + 2.0*a3z[i] + a4z[i]);
}

void runge_kutta_4_step_gpu(Bodies& bodies, Bodies& k1, Bodies& k2, Bodies& k3, Bodies& k4, T tau) {
    int n = bodies.n;
    int blocks = (n + block_size - 1) / block_size;
    T tau_2 = tau / 2.0;
    T tau_6 = tau / 6.0;
    
    compute_acceleration_kernel<<<blocks, block_size>>>(
        bodies.m, bodies.rx, bodies.ry, bodies.rz,
        k1.ax, k1.ay, k1.az, n);
//    CUDA_CHECK(cudaGetLastError());

	cudaMemcpy(k1.vx, bodies. vx, n * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(k1.vy, bodies.vy, n * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(k1.vz, bodies.vz, n * sizeof(T), cudaMemcpyDeviceToDevice);

//    CUDA_CHECK(cudaGetLastError());

    runge_kutta_4_k<<<blocks, block_size>>>(
        bodies.rx, bodies. ry, bodies. rz,
        bodies.vx, bodies.vy, bodies.vz,
        k1.vx, k1.vy, k1.vz,
        k1.ax, k1.ay, k1.az,
        k2.rx, k2.ry, k2.rz,
        k2.vx, k2.vy, k2.vz,
        tau_2, n);

    compute_acceleration_kernel<<<blocks, block_size>>>(
        bodies.m, k2.rx, k2.ry, k2.rz,
        k2.ax, k2.ay,  k2.az, n);

    runge_kutta_4_k<<<blocks, block_size>>>(
        bodies.rx, bodies.ry, bodies.rz,
        bodies.vx, bodies.vy, bodies.vz,
        k2.vx, k2.vy, k2.vz,
        k2.ax, k2.ay, k2.az,
        k3.rx, k3.ry,  k3.rz,
        k3.vx, k3.vy, k3.vz,
        tau_2, n);

    compute_acceleration_kernel<<<blocks, block_size>>>(
        bodies.m, k3.rx, k3.ry, k3.rz,
        k3.ax, k3.ay,  k3.az, n);

    runge_kutta_4_k<<<blocks, block_size>>>(
        bodies.rx, bodies.ry, bodies.rz,
        bodies.vx, bodies.vy, bodies.vz,
        k3.vx, k3.vy, k3.vz,
        k3.ax, k3.ay,  k3.az,
        k4.rx, k4.ry, k4.rz,
        k4.vx, k4.vy,  k4.vz,
        tau, n);

    compute_acceleration_kernel<<<blocks, block_size>>>(
        bodies.m, k4.rx,  k4.ry, k4.rz,
        k4.ax,  k4.ay, k4.az, n);

    runge_kutta_4_final_step<<<blocks, block_size>>>(
        bodies.rx, bodies. ry, bodies. rz,
        bodies.vx, bodies.vy, bodies.vz,
        k1.vx, k1.vy, k1.vz,  // v1
        k2.vx, k2.vy, k2.vz,  // v2
        k3.vx, k3.vy, k3.vz,  // v3
        k4.vx, k4.vy, k4.vz,  // v4
        k1.ax, k1.ay, k1.az,  // a1
        k2.ax, k2.ay,  k2.az,  // a2
        k3.ax,  k3.ay, k3.az,  // a3
        k4.ax, k4.ay, k4.az,  // a4
        tau_6, n);
}

int main()
{
    bool read = false;
    bool write = false;
	std::vector<std::ofstream> files;
	int num_steps = 160;
	T tau = 0.5/4.0;
    int n = 0;
	std::vector<T> m;
	std::vector<T> rx, ry, rz, vx, vy, vz;
    if (read) {
    	read_data("4body.txt", n, m, rx, ry, rz, vx, vy, vz);
    }
    else {
    	n = 35;
    	generate_random_bodies(n, m, rx, ry, rz, vx, vy, vz);
    }
	Bodies bodies;

	std::cout << "Number of bodies = " << n << std::endl;
	std::cout << "Number of steps: " << num_steps << std::endl;
	std::cout << "Size of block: " << block_size << std::endl;

	Bodies k1, k2, k3, k4;
	allocate_bodies_device(bodies, n);
	allocate_bodies_device(k1, n);
	allocate_bodies_device(k2, n);
	allocate_bodies_device(k3, n);
	allocate_bodies_device(k4, n);

	copy_to_device(bodies, m, rx, ry, rz, vx, vy, vz);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int step = 0; step < num_steps; ++step) {
		runge_kutta_4_step_gpu(bodies, k1, k2, k3, k4, tau);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    if (write) {
    	initialize_output_files(n, files);
    	write_positions(num_steps * tau, bodies, files);
    }

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Total time: " << milliseconds << " ms" << std::endl;
	std::cout << "Time per step: " << milliseconds / num_steps << " ms" << std::endl;

	free_bodies_device(bodies);
	free_bodies_device(k1);
	free_bodies_device(k2);
	free_bodies_device(k3);
	free_bodies_device(k4);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}