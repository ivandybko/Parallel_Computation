#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <omp.h>
#include <iomanip>

const double G = 6.67*1e-11;
const double eps = 1e-10;


auto time_copy = 0.0;
auto time_k2 = 0.0;
auto time_k3 = 0.0;
auto time_k4 = 0.0;
auto time_a = 0.0;
auto time_cycle = 0.0;

struct Body{
	double m = 0.0;
//	std::vector<double> r = std::vector<double>(3, 0.0);;
//	std::vector<double> v = std::vector<double>(3, 0.0);
//	std::vector<double> a = std::vector<double>(3, 0.0);
	double r[3] = {0.0, 0.0, 0.0};
	double v[3] = {0.0, 0.0, 0.0};
	double a[3] = {0.0, 0.0, 0.0};

};

void read_data(const std::string& filename, std::vector<Body>& bodies)
{
	std::ifstream file(filename);
	if (!file) {
		std::cout << "Не удалось открыть файл " << filename << "\n";
	}

	int N;
	file >> N;
	bodies.clear();

	for (int i = 0; i < N; ++i) {
		Body b;
		file >> b.m >> b.r[0] >> b.r[1] >> b.r[2] >> b.v[0] >> b.v[1] >> b.v[2];
		bodies.push_back(b);
	}
}

void initialize_output_files(const std::vector<Body>& bodies,
	std::vector<std::ofstream>& files)
{
	size_t n = bodies.size();
	files.resize(n);

	for (size_t i = 0; i < n; ++i) {
		std::string filename = "body" + std::to_string(i+1) + ".txt";

		files[i].open(filename);  // перезаписываем файл
		if (!files[i].is_open()) {
			throw std::runtime_error("Не удалось открыть файл" + filename);
		}

		files[i] << std::fixed << std::setprecision(16);
	}
}

void write_positions(double t,
	const std::vector<Body>& bodies,
	std::vector<std::ofstream>& files)
{
	size_t n = bodies.size();
	for (size_t i = 0; i < n; ++i) {
		files[i] << t << " "
				 << bodies[i].r[0] << " "
				 << bodies[i].r[1] << " "
				 << bodies[i].r[2] << "\n";
	}
}

std::vector<Body> generate_random_bodies(size_t n,
	double min_mass = 1e26, double max_mass = 1e30,
	double pos_min = -1e10, double pos_max = 1e10,
	double vel_min = -1e5, double vel_max = 1e5) {

	std::vector<Body> bodies;
	bodies.resize(n);


	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> mass_dist(min_mass, max_mass);
	std::uniform_real_distribution<double> pos_dist(pos_min, pos_max);
	std::uniform_real_distribution<double> vel_dist(vel_min, vel_max);

//	#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		Body b;
		b.m = mass_dist(gen);
		for (int dim = 0; dim < 3; ++dim) {
			b.r[dim] = pos_dist(gen);
			b.v[dim] = vel_dist(gen);
			b.a[dim] = 0.0;
		}
		bodies[i] = b;
	}

	return bodies;
}

void compute_acceleration(std::vector<Body>& bodies) {
	size_t n = bodies.size();
	const double eps2 = eps * eps;


	#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		bodies[i].a[0] = 0.0;
		bodies[i].a[1] = 0.0;
		bodies[i].a[2] = 0.0;
	}

	#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		double x_i = bodies[i].r[0], y_i = bodies[i].r[1], z_i = bodies[i].r[2];
		double a_0 = 0.0, a_1 = 0.0, a_2 =0.0;
		for (size_t j = 0; j < n; ++j) {
			double dx = bodies[j].r[0] - x_i;
			double dy = bodies[j].r[1] - y_i;
			double dz = bodies[j].r[2] - z_i;
			double delta = dx * dx + dy * dy + dz * dz + eps2 ;
			double multiplier = G * bodies[j].m / (delta * std::sqrt(delta));
			a_0 += multiplier * dx;
			a_1 += multiplier * dy;
			a_2 += multiplier * dz;
		}
		bodies[i].a[0] = a_0;
		bodies[i].a[1] = a_1;
		bodies[i].a[2] = a_2;
	}
}

void runge_kutta_4_step(std::vector<Body>& bodies, double tau) {
	int n = bodies.size();

	compute_acceleration(bodies);

	time_copy -= omp_get_wtime();
	std::vector<Body> k2 = bodies;
	std::vector<Body> k3 = bodies;
	std::vector<Body> k4 = bodies;

	time_copy += omp_get_wtime();

	time_k2 -= omp_get_wtime();

#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; ++i)
		for (int d = 0; d < 3; ++d) {
			k2[i].r[d] = bodies[i].r[d] + 0.5 * tau * bodies[i].v[d];
			k2[i].v[d] = bodies[i].v[d] + 0.5 * tau * bodies[i].a[d];
		}

	time_k2 += omp_get_wtime();

	time_a -= omp_get_wtime();

	compute_acceleration(k2);

	time_a += omp_get_wtime();

	time_k3 -= omp_get_wtime();

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; ++i)
		for (int d = 0; d < 3; ++d) {
			k3[i].r[d] = bodies[i].r[d] + 0.5 * tau * k2[i].v[d];
			k3[i].v[d] = bodies[i].v[d] + 0.5 * tau * k2[i].a[d];
		}


	time_k3 += omp_get_wtime();

	time_a -= omp_get_wtime();

	compute_acceleration(k3);

	time_a += omp_get_wtime();

	time_k4 -= omp_get_wtime();

#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; ++i) {
		for (int d = 0; d < 3; ++d) {
			k4[i].r[d] = bodies[i].r[d] + tau * k3[i].v[d];
			k4[i].v[d] = bodies[i].v[d] + tau * k3[i].a[d];
		}
	}

	time_k4 += omp_get_wtime();

	time_a -= omp_get_wtime();

	compute_acceleration(k4);

	time_a += omp_get_wtime();

	time_cycle -= omp_get_wtime();
#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; ++i)
		for (int d = 0; d < 3; ++d) {
			double v1 = bodies[i].v[d];
			double v2 = k2[i].v[d];
			double v3 = k3[i].v[d];
			double v4 = k4[i].v[d];

			double a1 = bodies[i].a[d];
			double a2 = k2[i].a[d];
			double a3 = k3[i].a[d];
			double a4 = k4[i].a[d];

			bodies[i].r[d] += (tau / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4);
			bodies[i].v[d] += (tau / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
		}

	time_cycle += omp_get_wtime();

}

int main()
{
	bool random{true};
	std::vector<Body> bodies;
	if (random){
		auto time_gen = -omp_get_wtime();
		int num_bodies = 10'000;
		bodies = generate_random_bodies(num_bodies);
		std::cout << "Number of bodies = " << num_bodies << std::endl;
//		std::cout << "Number of threads = " << omp_get_num_procs() << std::endl;

		std::vector<Body> bodies_copy(bodies);
		time_gen += omp_get_wtime();
		std::cout << "Time for generation = " << time_gen << std::endl;
		double t = 0.0, tau = 0.01/4.0;
		int num_steps = 40;
//		int num_threads = 2;

		time_copy = 0;
		time_k2 = 0;
		time_k3 = 0;
		time_k4= 0;
		time_a = 0;
		time_cycle = 0;

		auto time_rk = -omp_get_wtime();
		for (int step = 0; step < num_steps; ++step) {
			runge_kutta_4_step(bodies, tau);
			t += tau;
		}
		time_rk += omp_get_wtime();

		std::cout << "Time of copying = " << time_copy << std::endl;
		std::cout << "Time of k2 = " << time_k2/num_steps << std::endl;
		std::cout << "Time of k3 = " << time_k3/num_steps << std::endl;
		std::cout << "Time of k4 = " << time_k4/num_steps << std::endl;
		std::cout << "Time of a = " << time_a/num_steps << std::endl;
		std::cout << "Time of cycle = " << time_cycle/num_steps << std::endl;


		std::cout << "Time of calculation = " << time_rk/num_steps << std::endl;
		std::cout << std::endl;

		omp_set_num_threads(1);



		time_copy = 0;
		time_k2 = 0;
		time_k3 = 0;
		time_k4= 0;
		time_a = 0;
		time_cycle = 0;

		auto time_rk_single = -omp_get_wtime();
		for (int step = 0; step < num_steps; ++step){
			runge_kutta_4_step(bodies_copy, tau);
			t += tau;
		}
		time_rk_single += omp_get_wtime();

		std::cout << "Time of copying = " << time_copy/num_steps << std::endl;
		std::cout << "Time of k2 = " << time_k2/num_steps << std::endl;
		std::cout << "Time of k3 = " << time_k3/num_steps << std::endl;
		std::cout << "Time of k4 = " << time_k4/num_steps << std::endl;
		std::cout << "Time of a = " << time_a/num_steps << std::endl;
		std::cout << "Time of cycle = " << time_cycle/num_steps << std::endl;


		std::cout << "Time of calculation (one kernel) = " << time_rk_single/num_steps << std::endl;
		std::cout << "Speedup = " << time_rk_single/time_rk << std::endl;


//		for (int i = 2; i <= 18; ++i){
//			omp_set_num_threads(i);
//			std::cout << "Number of threads = " << i << std::endl << std::endl;
//
//			std::vector<Body> bodies_amdal(bodies);
//
//
//			time_copy = 0;
//			time_k2 = 0;
//			time_k3 = 0;
//			time_k4= 0;
//			time_a = 0;
//			time_cycle = 0;
//
//			time_rk = -omp_get_wtime();
//			for (int step = 0; step < num_steps; ++step){
//				runge_kutta_4_step(bodies_amdal, tau);
//				t += tau;
//			}
//			time_rk += omp_get_wtime();
//
//			std::cout << "Time of copying = " << time_copy/num_steps << std::endl;
//			std::cout << "Time of k2 = " << time_k2/num_steps << std::endl;
//			std::cout << "Time of k3 = " << time_k3/num_steps << std::endl;
//			std::cout << "Time of k4 = " << time_k4/num_steps << std::endl;
//			std::cout << "Time of a = " << time_a/num_steps << std::endl;
//			std::cout << "Time of cycle = " << time_cycle/num_steps << std::endl;
//
//
//			std::cout << "Time of calculation (one kernel) = " << time_rk/num_steps << std::endl;
//			std::cout << "Speedup = " << time_rk_single/time_rk << std::endl;
//		}

	}
	else {
		omp_set_num_threads(4);
		read_data("4body.txt", bodies);
		std::vector<std::ofstream> files;
		initialize_output_files(bodies, files);
		double t = 0.0, tau = 0.01/8.0;
		int num_steps = 20.0 / tau + 1 ;
		int output_every = int(0.1 / tau);
		for (int step = 0; step < num_steps; ++step) {
			if (step % output_every < 1e-6){
				write_positions(t, bodies, files);

			}
			runge_kutta_4_step(bodies, tau);

			t += tau;
		}

	}
	return 0;
}