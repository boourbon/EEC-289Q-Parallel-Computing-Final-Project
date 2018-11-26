//EEC 289Q final project
//Team information: Huian Wang, Minhui Huang
//Solve Car rental problem in "Reinforcement Learning: An Introduction" using dynamic programming on GPU
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <algorithm>
#include <time.h>

//Global Parameter Initialization

//maximum # of cars in each location
int MAX_CARS = 20;

//maximum # of cars to move during night
int MAX_MOVE_OF_CARS = 5;

//expectation for rental requests in first location
int RENTAL_REQUEST_FIRST_LOC = 3;

//expectation for rental requests in second location
int RENTAL_REQUEST_SECOND_LOC = 4;

//expectation for # of cars returned in first location
int RETURNS_FIRST_LOC = 3;

//expectation for # of cars returned in second location
int RETURNS_SECOND_LOC = 2;

float DISCOUNT = 0.9;

//credit earned by a car
int RENTAL_CREDIT = 10;

//cost of moving a car
int MOVE_CAR_COST = 2;

// An up bound for poisson distribution
//If n is greater than this value, then the probability of getting n is truncated to 0
int POISSON_UP_BOUND = 11;

//compute factorial
long factorial(int n) {
	if (n <= 0) {
		return 1;
	}
	else {
		return n * factorial(n - 1);
	}
}


// state_x: # of cars in first location, state_y: # of cars in second location
// action: positive if moving cars from first location to second location, negative if moving cars from second location to first location
// stateValue: state value matrix
float expectedReturn_CPU(int state_x, int state_y, int action, float *stateValue, float *poisson) {
	int numOfCarsFirstLoc, numOfCarsSecondLoc, realRentalFirstLoc, realRentalSecondLoc, numOfCarsFirstLoc_, numOfCarsSecondLoc_;
	float reward, prob, prob_;
	
	int rentalRequestFirstLoc,rentalRequestSecondLoc, returnedCarsFirstLoc,returnedCarsSecondLoc;

	//initailize total return
	float returns = 0.0;

	//cost for moving cars
	returns = returns - MOVE_CAR_COST * abs(action);

	// go through all possible rental requests
	for ( rentalRequestFirstLoc = 0; rentalRequestFirstLoc < POISSON_UP_BOUND; rentalRequestFirstLoc++) {
		for ( rentalRequestSecondLoc = 0; rentalRequestSecondLoc < POISSON_UP_BOUND; rentalRequestSecondLoc++) {
			numOfCarsFirstLoc = std::min(state_x - action, MAX_CARS);
			numOfCarsSecondLoc = std::min(state_y + action, MAX_CARS);

			// valid rental requests should be less than actual # of cars
			realRentalFirstLoc = std::min(numOfCarsFirstLoc, rentalRequestFirstLoc);
			realRentalSecondLoc = std::min(numOfCarsSecondLoc, rentalRequestSecondLoc);

			// get credits for renting
			reward = (realRentalFirstLoc + realRentalSecondLoc) * RENTAL_CREDIT;
			numOfCarsFirstLoc = numOfCarsFirstLoc - realRentalFirstLoc;
			numOfCarsSecondLoc = numOfCarsSecondLoc - realRentalSecondLoc;

			// probability for current combination of rental requests
			prob = poisson[0*POISSON_UP_BOUND + rentalRequestFirstLoc] * poisson[1*POISSON_UP_BOUND + rentalRequestSecondLoc];

			//record # of cars in each location and prob
			numOfCarsFirstLoc_ = numOfCarsFirstLoc;
			numOfCarsSecondLoc_ = numOfCarsSecondLoc;
			prob_ = prob;

			//consider the returned cars case
			for ( returnedCarsFirstLoc = 0; returnedCarsFirstLoc < POISSON_UP_BOUND; returnedCarsFirstLoc++) {
				for ( returnedCarsSecondLoc = 0; returnedCarsSecondLoc < POISSON_UP_BOUND; returnedCarsSecondLoc++) {
					numOfCarsFirstLoc = numOfCarsFirstLoc_;
					numOfCarsSecondLoc = numOfCarsSecondLoc_;
					prob = prob_;
					numOfCarsFirstLoc = std::min(numOfCarsFirstLoc + returnedCarsFirstLoc, MAX_CARS);
					numOfCarsSecondLoc = std::min(numOfCarsSecondLoc + returnedCarsSecondLoc, MAX_CARS);
					prob = prob * poisson[2*POISSON_UP_BOUND + returnedCarsFirstLoc] * poisson[3*POISSON_UP_BOUND + returnedCarsSecondLoc];
					returns = returns + prob * (reward + DISCOUNT * stateValue[numOfCarsFirstLoc*(MAX_CARS + 1) + numOfCarsSecondLoc]);
				}
			}
		}
	}
									
	return returns;
}

void print_matrix(int *matrix, int n, int m){
	int i,j;

	for(i = 0; i < n; i++){
		for (j = 0; j < m; j++){
			printf("%d   ", matrix[i*m + j]);
		}
		printf("\n");
	}

	printf("\n");
}


int main() {
	//current policy and state value
	int policy[MAX_CARS+1][MAX_CARS+1];
	//int newPolicy[MAX_CARS+1][MAX_CARS+1];
	float stateValue[MAX_CARS+1][MAX_CARS+1], oldStateValue[MAX_CARS+1][MAX_CARS+1], v;
	//float newStateValue[MAX_CARS+1][MAX_CARS+1];
	int i, j, action, old_action;

	for (i = 0; i <= MAX_CARS; i++) {
		for (j = 0; j <= MAX_CARS; j++) {
			policy[i][j] = 0;
			//newPolicy[i][j] = 0;
			stateValue[i][j] = 0;
			//newStateValue[i][j] = 0;
		}
	}
	

	// Probability for poisson distribution
	float poisson[4][POISSON_UP_BOUND];

	for (i = 0; i < POISSON_UP_BOUND; i++) {
		poisson[0][i] = exp(-RENTAL_REQUEST_FIRST_LOC) * pow(RENTAL_REQUEST_FIRST_LOC, i) / factorial(i);
		poisson[1][i] = exp(-RENTAL_REQUEST_SECOND_LOC) * pow(RENTAL_REQUEST_SECOND_LOC, i) / factorial(i);
		poisson[2][i] = exp(-RETURNS_FIRST_LOC) * pow(RETURNS_FIRST_LOC, i) / factorial(i);
		poisson[3][i] = exp(-RETURNS_SECOND_LOC) * pow(RETURNS_SECOND_LOC, i) / factorial(i);
	}

	//printf("%f\n", poisson[0][0]);

	//policy iteration
	bool policy_stable = false;
	//int policyImprovementInd = 0;

	clock_t start, finish; 
	double Total_time; 

	start = clock(); 

	float tolerance = 0.0001;
	while (policy_stable == false) {
		
		//Policy iteration
		while(true){
			double delta = 0;

			for (i = 0; i <= MAX_CARS; i++)
				for (j = 0; j <= MAX_CARS; j++)
					oldStateValue[i][j] = stateValue[i][j];
			for (i = 0; i <= MAX_CARS; i++){
				for (j = 0; j <= MAX_CARS; j++){
					v = stateValue[i][j];
					stateValue[i][j] = expectedReturn_CPU(i,j,policy[i][j], *oldStateValue, *poisson);
					delta = std::max(delta, fabs(v - stateValue[i][j]));
				}
			}

			if (delta < tolerance){
				//printf("%f\n", delta);
				break;
			}
		}


		//Policy inprovement
		policy_stable = true;
		for (i = 0; i <= MAX_CARS; i++){
			for (j = 0; j <= MAX_CARS; j++){
				old_action = policy[i][j];
				float temp = - 1.0e38;

				for(action = - MAX_MOVE_OF_CARS; action <= MAX_MOVE_OF_CARS; action ++){
					if (action <= i && action >=-j){
						float temp_return = expectedReturn_CPU(i, j, action, *stateValue, *poisson);
						if (temp_return > temp){
							temp = temp_return;
							policy[i][j] = action;
						}
					}
				}

				if(old_action != policy[i][j]){
					policy_stable = false;
				}
			}
		}

		print_matrix(*policy, MAX_CARS+1, MAX_CARS+1);

	}

	finish = clock(); 

	Total_time = (double)(finish-start) / CLOCKS_PER_SEC; 

	printf("Running time: %f seconds\n",Total_time);

	return 0;
	
}
