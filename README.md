# Parallel Computations

This repository contains laboratory works on parallel computing using OpenMP, MPI, and CUDA technologies.

## Contents

- [Lab 1: LU Decomposition with OpenMP](#lab-1-lu-decomposition-with-openmp)
- [Lab 2: N-body Problem with OpenMP](#lab-2-n-body-problem-with-openmp)
- [Lab 3: Helmholtz Equation with MPI](#lab-3-helmholtz-equation-with-mpi)
- [Lab 4: N-body Problem with MPI + OpenMP](#lab-4-n-body-problem-with-mpi)
- [Lab 5: N-body Problem with CUDA](#lab-5-n-body-problem-with-cuda)
- [Building and Running](#building-and-running)

---

## Lab 1: LU Decomposition with OpenMP

### Problem Description

Implementation of block LU decomposition using Demmel's algorithm with parallelization via OpenMP.

### Solution

- **Technology:** OpenMP
- **File:** `Lab 1/LU_Dybko.cpp`
- **Algorithm:** Block LU decomposition using Demmel's method
- **Matrix size test:** 8192×8192
- **Block size test:** 64×64

**Key features:**
- Parallelization using OpenMP directives
- Block processing for improved data locality
- Correctness verification through residual computation

**Results:**
- Speedup graphs plotted against number of threads

---

## Lab 2: N-body Problem with OpenMP

### Problem Description

Simulation of gravitational interaction between N bodies in three-dimensional space using the 4th-order Runge-Kutta method.

### Solution

- **Technology:** OpenMP
- **File:** `Lab 2/N_body_Dybko.cpp`
- **Method:** 4th-order Runge-Kutta (RK4)
- **Order of convergence:** 4
- **Gravitational constant:** G = 6.67×10⁻¹¹

**Mathematical formulation:**

The motion of bodies is described by Newton's equations:

$$m_i \frac{d^2\vec{r}_i}{dt^2} = \vec{F}_i$$

where the gravitational acceleration is calculated using Newton's law of universal gravitation:

$$\vec{a}_i = \sum_{j=1, j \neq i}^{N} \frac{Gm_j(\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}$$

For numerical stability, a small positive parameter $\varepsilon$ is introduced:

$$\vec{a}_i = \sum_{j=1}^{N} \frac{Gm_j(\vec{r}_j - \vec{r}_i)}{(|\vec{r}_j - \vec{r}_i|^2 + \varepsilon^2)^{3/2}}$$

The system is integrated using the RK4 method with time step $\Delta t$, providing 4th-order accuracy: error $\propto (\Delta t)^4$.

**Key features:**
- `Body` data structure storing mass, position, velocity, and acceleration
- Parallelization of gravitational force calculations between all pairs of bodies
- RK4 method for numerical integration of equations of motion
- Trajectory saving to separate files

**Order of convergence:** 4 (using 4th-order Runge-Kutta method)

**Results:**
- Speedup graphs for various thread counts
- Analysis of execution time for individual computational stages

---

## Lab 3: Helmholtz Equation with MPI

### Problem Description

Solving the two-dimensional Helmholtz equation using the Jacobi method with MPI for distributed computing.

### Solution

- **Technology:** MPI + OpenMP (hybrid model)
- **File:** `Lab 3/Helmholtz_Dybko.cpp`
- **Method:** Jacobi method with iterative refinement
- **Grid size:** 2001×2001
- **Tolerance:** ε = 10⁻⁸

**Equation:**

$$-\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} + k^2 u = f(x,y)$$

where the right-hand side is:

$$f(x,y) = 2\sin(\pi y) + k^2(1-x)x\sin(\pi y) + \pi^2(1-x)x\sin(\pi y)$$

**Boundary conditions:**

$$u(0,y) = u(1,y) = u(x,0) = u(x,1) = 0$$

**Analytical solution:**

$$u(x,y) = (1-x)x\sin(\pi y)$$

**Finite difference scheme:**

At each internal node $(i,j)$:

$$-\frac{y_{i+1,j} - 2y_{i,j} + y_{i-1,j}}{h^2} - \frac{y_{i,j+1} - 2y_{i,j} + y_{i,j-1}}{h^2} + k^2 y_{i,j} = f_{i,j}$$

**Jacobi iteration formula:**

$$y_{i,j}^{(n+1)} = \frac{1}{4 + k^2h^2}\left(y_{i+1,j}^{(n)} + y_{i-1,j}^{(n)} + y_{i,j+1}^{(n)} + y_{i,j-1}^{(n)} + h^2 f_{i,j}\right)$$

**Communication types:**

**3 different communication types** implemented for exchanging boundary values between processes:

1. **Send + Recv** (send_type = 1)
   - Blocking operation

2. **Sendrecv** (send_type = 2)
   - Blocking simultaneous data exchange
   - Simplified communication model

3. **Send_init + Start + Wait** (send_type = 3)
   - Non-blocking communication operations
   - Initialize once, use multiple times
   - Allows overlapping of computation and communication
   - **Faster when the number of points per thread is small**, as it reduces communication initialization overhead

**Results:**
- Speedup graphs for various numbers of MPI processes

---

## Lab 4: N-body Problem with MPI

### Problem Description

Distributed simulation of gravitational interaction between N bodies using MPI for distributed-memory parallelization.

### Solution

- **Technology:** MPI + OpenMP (hybrid model)
- **File:** `Lab 4/N_body_MPI_Dybko.cpp`
- **Method:** 4th-order Runge-Kutta (RK4)
- **Order of convergence:** 4

**Mathematical formulation:**

Same equations as Lab 2:

$$\vec{a}_i = \sum_{j=1}^{N} \frac{Gm_j(\vec{r}_j - \vec{r}_i)}{(|\vec{r}_j - \vec{r}_i|^2 + \varepsilon^2)^{3/2}}$$

with $G = 6.67 \times 10^{-11}$ m³/(kg·s²)

**Key features:**
- Distribution of bodies among MPI processes
- Hybrid parallelization (MPI for inter-node communication + OpenMP within node)
- Periodic data gathering for trajectory recording
- MPI collective operations for data synchronization

**Order of convergence:** 4 (4th-order Runge-Kutta method)

**Results:**
- Speedup graphs for various numbers of MPI processes

---

## Lab 5: N-body Problem with CUDA

### Problem Description

Simulation of gravitational interaction between N bodies on GPU using CUDA for massively parallel computations.

### Solution

- **Technology:** CUDA
- **File:** `Lab 5/N_body_CUDA_Dybko.cu`
- **Method:** 4th-order Runge-Kutta (RK4)
- **Order of convergence:** 4
- **Gravitational constant:** G = 6.67×10⁻¹¹
- **Block size:** 32×32
- **Data type:** double or float (for GPU performance optimization)

**Mathematical formulation:**

Same gravitational acceleration formula:

$$\vec{a}_i = \sum_{j=1}^{N} \frac{Gm_j(\vec{r}_j - \vec{r}_i)}{(|\vec{r}_j - \vec{r}_i|^2 + \varepsilon^2)^{3/2}}$$

**Key features:**
- `Bodies` data structure for storing arrays on GPU
- Kernel functions for computing gravitational accelerations on GPU
- Memory optimization and efficient global memory access patterns

**Order of convergence:** 4 (4th-order Runge-Kutta method)

---

## Building and Running

### Requirements

- **Compiler:** Clang icpc with OpenMP support
- **MPI:** OpenMPI or MPICH
- **CUDA:** CUDA Toolkit (for Lab 5)
- **CMake:** version 3.30 or higher

### Building

#### Labs 1-2 (OpenMP)

```bash
cd "Lab 1"  # or "Lab 2"
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

#### Labs 3-4 (MPI)

```bash
cd "Lab 3"  # or "Lab 4"
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make
```

#### Lab 5 (CUDA)

```bash
cd "Lab 5"
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
---

## Performance Results and Analysis

For laboratory works 1-4, **speedup graphs** were constructed as a function of parallelization parameters:
- **Lab 1:** speedup vs number of OpenMP threads
- **Lab 2:** speedup vs number of OpenMP threads
- **Lab 3:** speedup vs number of MPI processes 
- **Lab 4:** speedup vs number of MPI processes
The graphs demonstrate parallelization efficiency and allow evaluation of the scalability of the implemented algorithms.
