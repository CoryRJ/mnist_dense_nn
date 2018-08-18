#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "TRI/to_read.h"
#include "nn_class/Cnn.h"
#define SIZE 2
#define PRE 2
#define OUT 10
using namespace std;

float x_s(float f)
{
/*
	float v = (exp(f))/(1+exp(f));
	if(isnan(v))
	{
		if(signbit(f))
			return 0.0f;
		return 1.0f;
	}
	return v;	
*/
	if(f > 0)
		return f;
	return 0;
}

int main(int argc, char *argv[])
{	

	float image[784];

	get_n("mnist_dataset/train-images.idx3-ubyte",stoi(argv[1]),image);
	int dims[SIZE];
	float *result;
	dims[0] = 200;
	dims[1] = 200;
	dims[2] = 200;
	dims[3] = 200;
	dims[4] = 200;
	for(int i = 0; i < SIZE; i++)
	{
		dims[i] = 2;
	}
	dims[SIZE-1] = OUT;
	for(int i = 0; i <28*28; i++)
	{
//		image[i] = image[i]/100.0f;
		image[i] = 1;
	}

	Cnn aNet(SIZE,dims);
	aNet.setAct(&x_s);
	aNet.run(image);
	result = aNet.results();
	float sum = 0;
	for(int i = 0; i < OUT; i++)
	{
		cout << result[i] << " ";
		sum += result[i];
	}
	cout <<"= " << sum << endl;
	for(int i = 0; i < 100; i++)
	{
		aNet.backprop();
		aNet.run(image);
		result = aNet.results();
		for(int i = 0; i < OUT; i++)
		{
			cout << setprecision(PRE) << result[i] << " ";
		}
		cout << endl;
		cout << endl;
	}
		result = aNet.results();
	for(int i = 0; i < OUT; i++)
	{
		cout << result[i] << " ";
	}
	cout << endl;
	cout << get_n_result("mnist_dataset/train-labels.idx1-ubyte",stoi(argv[1])) << endl;
	free(result);
	return 0;
}
