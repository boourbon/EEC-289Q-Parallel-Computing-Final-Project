//EEC 289Q final project
//Team information: Huian Wang, Minhui Huang
//Solve Car rental problem in "Reinforcement Learning: An Introduction" using dynamic programming on GPU

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Global Parameter Initialization

//maximum # of cars in each location
#define MAX_CARS 20
//maximum # of cars to move during night
#define MAX_MOVE_OF_CARS 5
//expectation for rental requests in first location
#define RENTAL_REQUEST_FIRST_LOC 3
//expectation for rental requests in second location
#define RENTAL_REQUEST_SECOND_LOC 4
//expectation for # of cars returned in first location
#define RETURNS_FIRST_LOC 3
//expectation for # of cars returned in second location
#define RETURNS_SECOND_LOC 2
//discount
#define DISCOUNT 0.9
//credit earned by a car
#define RENTAL_CREDIT 10
//cost of moving a car
#define MOVE_CAR_COST 2
// An up bound for poisson distribution
//If n is greater than this value, then the probability of getting n is truncated to 0
#define POISSON_UP_BOUND 11

//compute factorial
long factorial(int n) {
	if (n <= 0) {
		return 1;
	}
	else {
		return n * factorial(n - 1);
	}
}

//print matrix
void print_matrix(float *matrix, int n, int m) {
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf("%f   ", matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_matrix_int(int *matrix, int n, int m) {
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf("%d   ", matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");
}


__global__
void Policy_Calculation_Kernel(float *returns, int *policy, float *stateValue, float *poisson) {

	//blockIdx.x = first_place, blockIdx.y = second_place
	//threadIdx.x = rentalRequestFirstLoc, threadIdx.y = rentalRequestSeecondLoc

	int numOfCarsFirstLoc, numOfCarsSecondLoc, realRentalFirstLoc, realRentalSecondLoc, numOfCarsFirstLoc_, numOfCarsSecondLoc_;
	float reward, prob, prob_;
	int rentalRequestFirstLoc, rentalRequestSecondLoc, returnedCarsFirstLoc, returnedCarsSecondLoc;

	__shared__ float s_stateValue[(MAX_CARS + 1)*(MAX_CARS + 1)];

	for (int i=threadIdx.x; i<=MAX_CARS; i=i+POISSON_UP_BOUND) {
        for (int j=threadIdx.y; j<=MAX_CARS; j=j+POISSON_UP_BOUND) {
            s_stateValue[i*(MAX_CARS + 1) + j] = stateValue[i*(MAX_CARS + 1) + j];
        }
    }
    __syncthreads();

    __shared__ float s_poisson[4 * POISSON_UP_BOUND];

    int a = threadIdx.x;
    if (a<=3) {
        s_poisson[threadIdx.x*POISSON_UP_BOUND + threadIdx.y] = poisson[threadIdx.x*POISSON_UP_BOUND + threadIdx.y];
    }
    __syncthreads();

    __shared__ float s_returns[POISSON_UP_BOUND*POISSON_UP_BOUND];
    
    s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y]= 0;

	//cost for moving cars
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] - MOVE_CAR_COST * abs(policy[blockIdx.x*(MAX_CARS + 1) + blockIdx.y]);
		//printf("MOVE_CAR_COST %f \n", returns[blockIdx.x*(MAX_CARS + 1) + blockIdx.y]);
	}

	//printf("%f \n", returns[blockIdx.x*(MAX_CARS + 1) + blockIdx.y]);
	numOfCarsFirstLoc = min(blockIdx.x - policy[blockIdx.x*(MAX_CARS + 1) + blockIdx.y], MAX_CARS);
	numOfCarsSecondLoc = min(blockIdx.y + policy[blockIdx.x*(MAX_CARS + 1) + blockIdx.y], MAX_CARS);
	//printf("%d \n", numOfCarsFirstLoc);
	rentalRequestFirstLoc = threadIdx.x;
	rentalRequestSecondLoc = threadIdx.y;

	// valid rental requests should be less than actual # of cars
	realRentalFirstLoc = min(numOfCarsFirstLoc, rentalRequestFirstLoc);
	realRentalSecondLoc = min(numOfCarsSecondLoc, rentalRequestSecondLoc);

	// get credits for renting
	reward = (realRentalFirstLoc + realRentalSecondLoc) * RENTAL_CREDIT;
	numOfCarsFirstLoc = numOfCarsFirstLoc - realRentalFirstLoc;
	numOfCarsSecondLoc = numOfCarsSecondLoc - realRentalSecondLoc;

	// probability for current combination of rental requests
	prob = s_poisson[0 * POISSON_UP_BOUND + rentalRequestFirstLoc] * s_poisson[1 * POISSON_UP_BOUND + rentalRequestSecondLoc];

	//record # of cars in each location and prob
	numOfCarsFirstLoc_ = numOfCarsFirstLoc;
	numOfCarsSecondLoc_ = numOfCarsSecondLoc;
	prob_ = prob;

	//consider the returned cars case
	for (returnedCarsFirstLoc = 0; returnedCarsFirstLoc < POISSON_UP_BOUND; returnedCarsFirstLoc++) {
		for (returnedCarsSecondLoc = 0; returnedCarsSecondLoc < POISSON_UP_BOUND; returnedCarsSecondLoc++) {
			numOfCarsFirstLoc = numOfCarsFirstLoc_;
			numOfCarsSecondLoc = numOfCarsSecondLoc_;
			prob = prob_;
			numOfCarsFirstLoc = min(numOfCarsFirstLoc + returnedCarsFirstLoc, MAX_CARS);
			numOfCarsSecondLoc = min(numOfCarsSecondLoc + returnedCarsSecondLoc, MAX_CARS);
			prob = prob * s_poisson[2 * POISSON_UP_BOUND + returnedCarsFirstLoc] * s_poisson[3 * POISSON_UP_BOUND + returnedCarsSecondLoc];
			
			s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] + prob * (reward + DISCOUNT * s_stateValue[numOfCarsFirstLoc*(MAX_CARS + 1) + numOfCarsSecondLoc]);
		}
	}
	returns[(blockIdx.x*POISSON_UP_BOUND+threadIdx.x)*(MAX_CARS + 1)*POISSON_UP_BOUND + blockIdx.y*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y];
	//printf("%d %d %d %d %f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, returns[(blockIdx.x*POISSON_UP_BOUND+threadIdx.x)*(MAX_CARS + 1)*POISSON_UP_BOUND + blockIdx.y*POISSON_UP_BOUND+threadIdx.y]);
}



__global__
void Policy_Calculation_Kernel2(float *returns, int action, float *stateValue, float *poisson) {

	//blockIdx.x = first_place, blockIdx.y = second_place
	//threadIdx.x = rentalRequestFirstLoc, threadIdx.y = rentalRequestSeecondLoc

	int numOfCarsFirstLoc, numOfCarsSecondLoc, realRentalFirstLoc, realRentalSecondLoc, numOfCarsFirstLoc_, numOfCarsSecondLoc_;
	float reward, prob, prob_;
	int rentalRequestFirstLoc, rentalRequestSecondLoc, returnedCarsFirstLoc, returnedCarsSecondLoc;

	__shared__ float s_stateValue[(MAX_CARS + 1)*(MAX_CARS + 1)];

	for (int i=threadIdx.x; i<=MAX_CARS; i=i+POISSON_UP_BOUND) {
        for (int j=threadIdx.y; j<=MAX_CARS; j=j+POISSON_UP_BOUND) {
            s_stateValue[i*(MAX_CARS + 1) + j] = stateValue[i*(MAX_CARS + 1) + j];
        }
    }
    __syncthreads();

    __shared__ float s_poisson[4 * POISSON_UP_BOUND];
    
    int c = threadIdx.x;
    if (c<=3) {
        s_poisson[threadIdx.x*POISSON_UP_BOUND + threadIdx.y] = poisson[threadIdx.x*POISSON_UP_BOUND + threadIdx.y];
    }
    __syncthreads();

    __shared__ float s_returns[POISSON_UP_BOUND*POISSON_UP_BOUND];
    
    s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y]= 0;

	//printf("%d \n", action);
	int a=blockIdx.x, b=blockIdx.y;
	if (action <= a && (-action) <= b) {
		//printf("%d \n", action);
		//cost for moving cars
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] - MOVE_CAR_COST * abs(action);
		}

		numOfCarsFirstLoc = min(blockIdx.x - action, MAX_CARS);
		numOfCarsSecondLoc = min(blockIdx.y + action, MAX_CARS);

		rentalRequestFirstLoc = threadIdx.x;
		rentalRequestSecondLoc = threadIdx.y;
		//printf("%d \n", action);
		// valid rental requests should be less than actual # of cars
		realRentalFirstLoc = min(numOfCarsFirstLoc, rentalRequestFirstLoc);
		realRentalSecondLoc = min(numOfCarsSecondLoc, rentalRequestSecondLoc);

		// get credits for renting
		reward = (realRentalFirstLoc + realRentalSecondLoc) * RENTAL_CREDIT;
		numOfCarsFirstLoc = numOfCarsFirstLoc - realRentalFirstLoc;
		numOfCarsSecondLoc = numOfCarsSecondLoc - realRentalSecondLoc;

		// probability for current combination of rental requests
		prob = s_poisson[0 * POISSON_UP_BOUND + rentalRequestFirstLoc] * s_poisson[1 * POISSON_UP_BOUND + rentalRequestSecondLoc];

		//record # of cars in each location and prob
		numOfCarsFirstLoc_ = numOfCarsFirstLoc;
		numOfCarsSecondLoc_ = numOfCarsSecondLoc;
		prob_ = prob;

		//consider the returned cars case
		for (returnedCarsFirstLoc = 0; returnedCarsFirstLoc < POISSON_UP_BOUND; returnedCarsFirstLoc++) {
			for (returnedCarsSecondLoc = 0; returnedCarsSecondLoc < POISSON_UP_BOUND; returnedCarsSecondLoc++) {
				numOfCarsFirstLoc = numOfCarsFirstLoc_;
				numOfCarsSecondLoc = numOfCarsSecondLoc_;
				prob = prob_;
				numOfCarsFirstLoc = min(numOfCarsFirstLoc + returnedCarsFirstLoc, MAX_CARS);
				numOfCarsSecondLoc = min(numOfCarsSecondLoc + returnedCarsSecondLoc, MAX_CARS);
				prob = prob * s_poisson[2 * POISSON_UP_BOUND + returnedCarsFirstLoc] * s_poisson[3 * POISSON_UP_BOUND + returnedCarsSecondLoc];
				s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y] + prob * (reward + DISCOUNT * s_stateValue[numOfCarsFirstLoc*(MAX_CARS + 1) + numOfCarsSecondLoc]);
			}
		}
		//printf("%d \n", action);
		//printf("INSIDE: %d %d %d %f \n", action, blockIdx.x, blockIdx.y, returns[(blockIdx.x*POISSON_UP_BOUND+threadIdx.x)*(MAX_CARS + 1)*POISSON_UP_BOUND + blockIdx.y*POISSON_UP_BOUND+threadIdx.y]);
	}
	returns[(blockIdx.x*POISSON_UP_BOUND+threadIdx.x)*(MAX_CARS + 1)*POISSON_UP_BOUND + blockIdx.y*POISSON_UP_BOUND+threadIdx.y] = s_returns[threadIdx.x*POISSON_UP_BOUND+threadIdx.y];
	//printf("INSIDE: %d %d %d %f \n", action, blockIdx.x, blockIdx.y, returns[(blockIdx.x*POISSON_UP_BOUND+threadIdx.x)*(MAX_CARS + 1)*POISSON_UP_BOUND + blockIdx.y*POISSON_UP_BOUND+threadIdx.y]);
}



int main() {
	float *returns;

	int *policy;

	float *stateValue;

	float *poisson;

	int old_action[(MAX_CARS + 1)*(MAX_CARS + 1)];

	float temp[(MAX_CARS + 1)*(MAX_CARS + 1)];

	int i, j, k, n, action;
	float v;

	size_t size1 = (MAX_CARS + 1)*(MAX_CARS + 1) * sizeof(float);
	size_t size2 = (MAX_CARS + 1)*(MAX_CARS + 1) * sizeof(int);
	size_t size3 = 4 * POISSON_UP_BOUND * sizeof(float);
	size_t size4 = (MAX_CARS + 1)*(MAX_CARS + 1)*  POISSON_UP_BOUND * POISSON_UP_BOUND * sizeof(float);
	cudaMallocManaged(&returns, size4);
	cudaMallocManaged(&policy, size2);
	cudaMallocManaged(&stateValue, size1);
	cudaMallocManaged(&poisson, size3);

	for (i = 0; i <= MAX_CARS; i++) {
		for (j = 0; j <= MAX_CARS; j++) {
			policy[i*(MAX_CARS + 1) + j] = 0;
			stateValue[i*(MAX_CARS + 1) + j] = 0;
			old_action[i*(MAX_CARS + 1) + j] = 0;
			temp[i*(MAX_CARS + 1) + j] = -1.0e38;
		}
	}

	for (i = 0; i < (MAX_CARS+1)*POISSON_UP_BOUND; i++) {
		for (j = 0; j < (MAX_CARS+1)*POISSON_UP_BOUND; j++) {
			returns[i*(MAX_CARS + 1)*POISSON_UP_BOUND + j] = 0;
		}
	}
	//print_matrix(stateValue, MAX_CARS + 1, MAX_CARS + 1);
	// Probability for poisson distribution

	for (i = 0; i < POISSON_UP_BOUND; i++) {
		poisson[0 * POISSON_UP_BOUND + i] = exp(-RENTAL_REQUEST_FIRST_LOC) * pow(RENTAL_REQUEST_FIRST_LOC, i) / factorial(i);
		poisson[1 * POISSON_UP_BOUND + i] = exp(-RENTAL_REQUEST_SECOND_LOC) * pow(RENTAL_REQUEST_SECOND_LOC, i) / factorial(i);
		poisson[2 * POISSON_UP_BOUND + i] = exp(-RETURNS_FIRST_LOC) * pow(RETURNS_FIRST_LOC, i) / factorial(i);
		poisson[3 * POISSON_UP_BOUND + i] = exp(-RETURNS_SECOND_LOC) * pow(RETURNS_SECOND_LOC, i) / factorial(i);
	}

	//printf("%f\n", poisson[0]);

	//policy iteration
	bool policy_stable = false;

	float tolerance = 0.0001;

	while (policy_stable == false) {

		//Policy iteration
		while (true) {

			float delta = 0;

			dim3 dimBlock(POISSON_UP_BOUND, POISSON_UP_BOUND);
			dim3 dimGrid(MAX_CARS + 1 , MAX_CARS + 1);
			Policy_Calculation_Kernel <<<dimGrid, dimBlock>>>(returns, policy, stateValue, poisson); //launch kernel

			cudaDeviceSynchronize();
			
			//print_matrix(returns, (MAX_CARS + 1)*POISSON_UP_BOUND, (MAX_CARS + 1)*POISSON_UP_BOUND);
			
			for (i = 0; i <= MAX_CARS; i++) {
				for (j = 0; j <= MAX_CARS; j++) {
					//stateValue[i*(MAX_CARS + 1) + j] = returns[i*(MAX_CARS + 1) + j];
					v = 0;
					for (k = 0; k < POISSON_UP_BOUND; k++) {
						for (n = 0; n < POISSON_UP_BOUND; n++) {
							v = v + returns[(i*POISSON_UP_BOUND + k)*(MAX_CARS + 1)*POISSON_UP_BOUND + j * POISSON_UP_BOUND + n];
							returns[(i*POISSON_UP_BOUND + k)*(MAX_CARS + 1)*POISSON_UP_BOUND + j * POISSON_UP_BOUND + n] = 0;
						}
					}
					
					delta = max(delta, abs(v - stateValue[i*(MAX_CARS + 1) + j]));
					stateValue[i*(MAX_CARS + 1) + j] = v;
				}
			}

			//print_matrix(stateValue, MAX_CARS + 1, MAX_CARS + 1);
			//break;
			//printf("%f\n", delta);

			if (delta < tolerance) {
				//printf("%f\n", delta);
				//print_matrix(stateValue, MAX_CARS + 1, MAX_CARS + 1);
				break;
			}
		}
		//break;

		//Policy inprovement GPU
		policy_stable = true;

		for (action = -MAX_MOVE_OF_CARS; action <= MAX_MOVE_OF_CARS; action++) {

			dim3 dimBlock(POISSON_UP_BOUND, POISSON_UP_BOUND);
			dim3 dimGrid(MAX_CARS + 1, MAX_CARS + 1);
			Policy_Calculation_Kernel2 <<<dimGrid, dimBlock>>>(returns, action, stateValue, poisson);

			cudaDeviceSynchronize();
			//printf("OUTSIDE\n");
			//print_matrix(returns, (MAX_CARS + 1)*POISSON_UP_BOUND, (MAX_CARS + 1)*POISSON_UP_BOUND);
			//printf("****************\n");

			for (i = 0; i <= MAX_CARS; i++) {
				for (j = 0; j <= MAX_CARS; j++) {
					v = 0;
					for (k = 0; k < POISSON_UP_BOUND; k++) {
						for (n = 0; n < POISSON_UP_BOUND; n++) {
							v = v + returns[(i*POISSON_UP_BOUND + k)*(MAX_CARS + 1)*POISSON_UP_BOUND + j * POISSON_UP_BOUND + n];
							returns[(i*POISSON_UP_BOUND + k)*(MAX_CARS + 1)*POISSON_UP_BOUND + j * POISSON_UP_BOUND + n] = 0;
						}
					}
					if (v > temp[i*(MAX_CARS + 1) + j]) {
						temp[i*(MAX_CARS + 1) + j] = v;
						policy[i*(MAX_CARS + 1) + j] = action;
					}
					
				}
			}

		}
		
		for (i = 0; i <= MAX_CARS; i++) {
			for (j = 0; j <= MAX_CARS; j++) {
				if (old_action[i*(MAX_CARS + 1) + j] != policy[i*(MAX_CARS + 1) + j]) {
					policy_stable = false;
					old_action[i*(MAX_CARS + 1) + j] = policy[i*(MAX_CARS + 1) + j];
				}
				temp[i*(MAX_CARS + 1) + j] = -1.0e38;
			}
		}

		print_matrix_int(policy, MAX_CARS + 1, MAX_CARS + 1);

	}

}
