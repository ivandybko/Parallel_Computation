#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include "mpi.h"


double f(double x, double y, double k) {
	return 2.0 * std::sin(M_PI * y) + k * k * (1.0 - x) * x * std::sin(M_PI * y) + M_PI * M_PI * (1.0 - x) * x * std::sin(M_PI * y);
}

int main()
{

	return 0;
}
