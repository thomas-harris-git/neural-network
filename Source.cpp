#include <Aria.h>
#include <iostream>
#include <math.h>
#include <fstream>


using namespace std;

float learning_rate = 0.005, alpha = 0.5, lamda = 0.6;
const int hidden_nodes = 2, train_data = 2087, val_data = 446, test_data = 446;
double Left_Motor, Right_Motor, sonar[7], sonarDistance[7];

double input_vector[3] = {};
double output_vector[2] = {};
double actual_output[2] = {};
double hidden_layer_vector[hidden_nodes + 1],
hidden_weights[hidden_nodes][3], output_weights[2][hidden_nodes + 1], error[2], output_gradient[2],
hidden_gradient[hidden_nodes + 1], delta_output_weights[2][hidden_nodes + 1], delta_hidden_weights[hidden_nodes][3],
train_data_input1[train_data], train_data_input2[train_data], train_data_output1[train_data], train_data_output2[train_data],
val_data_input1[train_data], val_data_input2[train_data], val_data_output1[train_data], val_data_output2[train_data],
test_data_input1[train_data], test_data_input2[train_data], test_data_output1[train_data], test_data_output2[train_data],
error_sum[2], train_RMSE[100], val_RMSE[100], test_RMSE[100];

double activation_function(double x) 
{
	return 1 / (1 + exp(x)); //sigmoid fuction
}

void initilise_weights()
{
	int i, j;
	for (i = 0; i < hidden_nodes; i++)
		for (j = 0; j < 3; j++)
			hidden_weights[i][j] = ((double)rand() / (RAND_MAX));

	for (i = 0; i < 2; i++)
		for (j = 0; j < hidden_nodes + 1; j++)
			output_weights[i][j] = ((double)rand() / (RAND_MAX));

	for (i = 0; i < hidden_nodes; i++)
		for (j = 0; j < 3; j++)
			delta_hidden_weights[i][j] = 0;

	for (i = 0; i < 2; i++)
		for (j = 0; j < hidden_nodes + 1; j++)
			delta_output_weights[i][j] = 0;
}

void import_data()
{
	/*
	Total number of rows in data file is 2981
	Training =   70%  2087  (0 - 2087)
	Validation = 15%  446  (2088 - 2534)
	Testing =    15%  446  (2535 - 2981)
	*/
	fstream file("nntrainingdata.csv");
	if (!file.is_open())
		printf("Error opening file");
	string in1, in2, out1, out2;
	int x = 0,y = 0,z = 0;
	while (file.good())
	{
		getline(file, in1, ',');
		getline(file, in2, ',');
		getline(file, out1, ',');
		getline(file, out2, '\n');

		if (in1 == "")
			break;

		if (x < 2088) {
			train_data_input1[x] = stod(in1);
			train_data_input2[x] = stod(in2);
			train_data_output1[x] = stod(out1);
			train_data_output2[x] = stod(out2);
		}
		if (x >= 2088 && x < 2535)
		{
			val_data_input1[y] = stod(in1);
			val_data_input2[y] = stod(in2);
			val_data_output1[y] = stod(out1);
			val_data_output2[y] = stod(out2);
			y++;
		}
		if (x >= 2535)
		{
			test_data_input1[z] = stod(in1);
			test_data_input2[z] = stod(in2);
			test_data_output1[z] = stod(out1);
			test_data_output2[z] = stod(out2);
			z++;
		}
		x++;
	}

	file.close();

	//printf("Input 1: %f Input 2: %f Output 1: %f Output 2: %f \n", train_data_input1[1], train_data_input2[1], train_data_output1[1], train_data_output2[1]);
}
void set_inputs_outputs(int x, int i){
	if (i == 0) {
		input_vector[0] = 1;
		input_vector[1] = (train_data_input1[x] - 0) / (5000 - 0);
		input_vector[2] = (train_data_input2[x] - 0) / (5000 - 0);
		actual_output[0] = (train_data_output1[x] - 100) / (400 - 100);
		actual_output[1] = (train_data_output2[x] - 100) / (400 - 100);
	}
	if (i == 1) {
		input_vector[0] = 1;
		input_vector[1] = (val_data_input1[x] - 0) / (5000 - 0);
		input_vector[2] = (val_data_input2[x] - 0) / (5000 - 0);
		actual_output[0] = (val_data_output1[x] - 100) / (400 - 100);
		actual_output[1] = (val_data_output2[x] - 100) / (400 - 100);
	}

	if (i == 2) {
		input_vector[0] = 1;
		input_vector[1] = (test_data_input1[x] - 0) / (5000 - 0);
		input_vector[2] = (test_data_input2[x] - 0) / (5000 - 0);
		actual_output[0] = (test_data_output1[x] - 100) / (400 - 100);
		actual_output[1] = (test_data_output2[x] - 100) / (400 - 100);
	}

	//printf("\nInput 1: %f Input 2: %f Output 1: %f Output 2: %f \n\n", input_vector[1], input_vector[2], actual_output[0], actual_output[1]);
}

void forward()
{
	int i, j, k;
	
	/*printf("Hidden Weights:\n");
	for (i = 0; i < hidden_nodes; i++) {
		for (j = 0; j < 3; j++)
			printf("%f ", hidden_weights[i][j]);
		printf("\n");
	}
	printf("\n");*/
	
	for (i = 0; i < hidden_nodes; i++)
		hidden_layer_vector[i] = 0;

	for (k = 0; k < 2; k++)
		output_vector[k] = 0;
	
	hidden_layer_vector[0] = 1; //Bias
	//printf("Hidden Layer Vector: \n %f ", hidden_layer_vector[0]);
	for (i = 0; i < hidden_nodes; i++) {
		for (j = 0; j < 3; j++)
		{
			hidden_layer_vector[i+1] += input_vector[j] * hidden_weights[i][j];
			
		}
		//printf("%f ", hidden_layer_vector[i+1]);
	}

	//printf("\n\n");
	
	//printf("Hidden Layer Vecotr After Activation Function: \n %f ", hidden_layer_vector[0]);
	for (i = 1; i < hidden_nodes+1; i++)
	{
		hidden_layer_vector[i] = activation_function(hidden_layer_vector[i]);
		//printf("%f ", hidden_layer_vector[i]);
	}

	/*printf("\n\n");
	printf("Output Weights:\n");
	for (i = 0; i < 2; i++) {
		for (j = 0; j < hidden_nodes + 1; j++)
			printf("%f ", output_weights[i][j]);
		printf("\n");
	}
	printf("\n");
	printf("Output Vector:\n");*/
	for (k = 0; k < 2; k++) {
		for (i = 0; i < hidden_nodes+1; i++)
		{
			output_vector[k] += hidden_layer_vector[i] * output_weights[k][i];
		}
		//printf("%f ", output_vector[k]);
	}

	//printf("\n\n");
	//printf("Output Vector after Activation Function\n");
	for (k = 0; k < 2; k++)
	{
		output_vector[k] = activation_function(output_vector[k]);
		//printf("%f ", output_vector[k]);
	}
	//printf("\n\n");
	//printf("Error\n");
	for (k = 0; k < 2; k++)
	{
		error[k] = output_vector[k] - actual_output[k];
		error_sum[k] = error_sum[k] + pow(error[k], 2);
		//printf("%f - %f = ", output_vector[k], actual_output[k]);
		//printf("%f\n", error[k]);
		//printf("Error Sum: %f\n", error_sum[k]);
	}
}

void backward()
{
	int k, i, j;

	//printf("\n");

	//printf("Output Gradient\n");
	for (k = 0; k < 2; k++)
	{
		output_gradient[k] = lamda * output_vector[k] * (1 - output_vector[k]) * error[k];
		//printf("%f ", output_gradient[k]);
	}

	//printf("\n\n");

	//printf("Delta Output Weights\n");
	for (k = 0; k < 2; k++){
		for (i = 0; i < hidden_nodes + 1; i++) {
			delta_output_weights[k][i] = learning_rate * output_gradient[k] * hidden_layer_vector[i] + alpha * delta_output_weights[k][i];
			//printf("%f ", delta_output_weights[k][i]);
		}
		//printf("\n");
	}

	//printf("\n");

	//printf("Hidden Gradient\n");
	for (i = 0; i < hidden_nodes; i++){
		for (k = 0; k < 2; k++) {
			hidden_gradient[i] += (output_gradient[k] * output_weights[k][i+1]);
		}
		hidden_gradient[i] += lamda * hidden_layer_vector[i+1] * (1 - hidden_layer_vector[i+1]);
		//printf("%f ", hidden_gradient[i]);
	}

	//printf("\n\n");
	//printf("Delta Hidden Weights\n");
	for (i = 0; i < hidden_nodes; i++) {
		for (j = 0; j < 3; j++)
		{
			delta_hidden_weights[i][j] = learning_rate * hidden_gradient[i] * input_vector[j] + alpha * delta_hidden_weights[i][j];
			//printf("%f ", delta_hidden_weights[i][j]);
		}
		//printf("\n");
	}
	//printf("\n");

	/*
	Update Weights
	*/
	//printf("Updated Hidden weights\n");
	for (i = 0; i < hidden_nodes; i++) {
		for (j = 0; j < 3; j++) {
			hidden_weights[i][j] = delta_hidden_weights[i][j] + hidden_weights[i][j];
			//printf("%f ", hidden_weights[i][j]);
		}
		//printf("\n");
	}
	//printf("\n");
	//printf("Updated Output weights\n");
	for (k = 0; k < 2; k++) {
		for (i = 0; i < hidden_nodes + 1; i++) {
			output_weights[k][i] = delta_output_weights[k][i] + output_weights[k][i];
			//printf("%f ", output_weights[k][i]);
		}
		//printf("\n");
	}
	//printf("\n");
	for (i = 0; i < hidden_nodes; i++)
	{
		hidden_gradient[i] = 0;
	}
}

void shuffle()
{
	int i;
	double train_data_input1_shuffled[train_data], train_data_input2_shuffled[train_data],
		train_data_output1_shuffled[train_data], train_data_output2_shuffled[train_data];
	vector<int> indexes;
	indexes.reserve(train_data);
	for (i = 0; i < train_data; ++i)
		indexes.push_back(i);
	random_shuffle(indexes.begin(), indexes.end());

	i = 0;
	for (vector<int>::iterator it = indexes.begin(); it != indexes.end(); ++it) {
		train_data_input1_shuffled[i] = train_data_input1[*it];
		train_data_input2_shuffled[i] = train_data_input2[*it];
		train_data_output1_shuffled[i] = train_data_output1[*it];
		train_data_output2_shuffled[i] = train_data_output2[*it];
		i++;
	}

	for (i = 0; i < train_data; ++i)
	{
		train_data_input1[i] = train_data_input1_shuffled[i];
		train_data_input2[i] = train_data_input2_shuffled[i];
		train_data_output1[i] = train_data_output1_shuffled[i];
		train_data_output2[i] = train_data_output2_shuffled[i];
	}

	/*printf("\n\n");

	for (i = 0; i < 10; ++i)
	{
		printf("%f ", train_data_input1[i]);
	}

	printf("\n");

	for (i = 0; i < 10; ++i)
	{
		printf("%f ", train_data_input2[i]);
	}

	printf("\n\n");*/
}

void print_weights()
{
	printf("Hidden Weights\n");
	for (int i = 0; i < hidden_nodes; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%f ", hidden_weights[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
	printf("Output Weights\n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < hidden_nodes + 1; j++) {
			printf("%f ", output_weights[i][j]);
		}
		printf("\n");
	}

	printf("\n\n");
}

void Root_Mean_Squared_Error()
{
	initilise_weights();
	import_data();

	double left_RMSE, right_RMSE;

	for (int x = 0; x < 100; x++) //26
	{
		for (int i = 0; i < train_data; i++) {
			set_inputs_outputs(i,0);
			forward();
			backward();
		}
		left_RMSE = sqrt(error_sum[0]/train_data);
		right_RMSE = sqrt(error_sum[1]/train_data);
		train_RMSE[x] = (left_RMSE + right_RMSE) / 2;
		shuffle();
		error_sum[0] = 0;
		error_sum[1] = 0;
		
		for (int i = 0; i < val_data; i++) {
			set_inputs_outputs(i,1);
			forward();
		}
		left_RMSE = sqrt(error_sum[0] / val_data);
		right_RMSE = sqrt(error_sum[1] / val_data);
		val_RMSE[x] = (left_RMSE + right_RMSE) / 2;
		error_sum[0] = 0;
		error_sum[1] = 0;

		for (int i = 0; i < test_data; i++) {
			set_inputs_outputs(i, 2);
			forward();
		}
		left_RMSE = sqrt(error_sum[0] / val_data);
		right_RMSE = sqrt(error_sum[1] / val_data);
		test_RMSE[x] = (left_RMSE + right_RMSE) / 2;
		error_sum[0] = 0;
		error_sum[1] = 0;
	}
	print_weights();
}

void write_to_file()
{
	ofstream myfile;
	myfile.open("output_data.csv");
	myfile << "Epochs,Train,Validation,Test\n";
	for (int x = 0; x < 100; x++)
	{
		myfile << x << "," << train_RMSE[x] << "," << val_RMSE[x] << "," << test_RMSE[x] <<  ",\n";
	}
	myfile.close();
}

void Network_Weights()
{
	hidden_weights[0][0] = 5.459407;
	hidden_weights[0][1] = 1.172272;
	hidden_weights[0][2] = 2.394207;
	hidden_weights[1][0] = 5.611499;
	hidden_weights[1][1] = 0.826171;
	hidden_weights[1][2] = 2.371549;

	output_weights[0][0] = 2.327857;
	output_weights[0][1] = 1.014490;
	output_weights[0][2] = 0.891365;
	output_weights[1][0] = 1.549120;
	output_weights[1][1] = 0.233026;
	output_weights[1][2] = 0.893642;

	//print_weights();
}

void set_motor_values()
{
	input_vector[0] = 1;
	input_vector[1] = (sonarDistance[0] - 0) / (5000 - 0);
	input_vector[2] = (sonarDistance[1] - 0) / (5000 - 0);
	forward();
	for (int k = 0; k < 2; k++)
	{
		output_vector[k] = output_vector[k] * (400 - 100) + 100;
	}
	Left_Motor = output_vector[0];
	Right_Motor = output_vector[1];
}

int main(int argc, char **argv)
{
	//Root_Mean_Squared_Error();
	//write_to_file();
	
	/*Network_Weights();
	for (int i = 0; i < 20; i++) {
		set_inputs_outputs(i, 2);
		forward();
		printf("Left Error: %f Right Error: %f \n", error[0], error[1]);
	}
	
	while (true){}*/
	
	// create instances 
	Aria::init();
	ArRobot robot;
	ArPose pose;
	// parse command line arguments 
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();

	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		std::cout << "Robot connected!" << std::endl;

	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();
	
	Network_Weights();

	while (true)
	{
		for (int i = 0; i < 8; i++)
		{
			sonar[i] = robot.getSonarReading(i)->getRange();
			if (0 < sonar[i] < 5000)
				sonarDistance[i] = sonar[i];
			if (sonarDistance[i] > 5000)
				sonarDistance[i] = 5000;
		}
		set_motor_values();
		robot.setVel2(Left_Motor, Right_Motor);
		//printf("Left: %f Right: %f \n", Left_Motor, Right_Motor);
		//ArUtil::sleep(500);
	}
	
	robot.lock();
	robot.stop();
	robot.unlock();
	// terminate all threads and exit 
	Aria::exit();
}