#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "TRI/to_read.h"
#include "nn_class/Cnn.h"
#define SIZE 5
#define PRE 2
#define OUT 10
#define IN 28*28
#define MID 28*28
#define TESTS 500
#define TESTROUNDS 25
#define BATCH 25
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

	cout << setprecision(PRE);
	float image[IN];
	int dims[SIZE];
	float *result;
	for(int i = 0; i < SIZE; i++)
	{
		dims[i] = MID;
	}
	dims[0] = IN;
	dims[SIZE-1] = OUT;

	Cnn aNet(SIZE,dims);
	aNet.setAct(&x_s);
	float *values = (float*) malloc(sizeof(float)*10);
	for(int i = 0; i < OUT; i++)
	{
		values[i] = 0;
	}
	int index = 0;
	int res = 0;
	for(int i = 0; i < TESTS; i++)
	{
		aNet.reset();
		cout << "Starting batch: " << i << " / " << TESTS << " ... ... ..." << endl;
		for(int j = 0; j < BATCH; j++)
		{
			
			get_n("mnist_dataset/train-images.idx3-ubyte",index,image);
			for(int l = 0; l <IN; l++)
			{
				image[l] = image[l]/100.0f;
			}
			res = get_n_result("mnist_dataset/train-labels.idx1-ubyte",index);
			values[res] = 1;
			aNet.run(image);
			aNet.backprop(values);
			values[res] = 0;
			index++;
		}
		cout << "Done! Updating... ... ..." << endl;
		aNet.update();
		int numBigger = 0;
		if(i%10 == 0)
		{
			cout << "Testing for: " << TESTROUNDS << endl;
			for(int j = 0; j < TESTROUNDS; j++)
			{
				
				get_n("mnist_dataset/train-images.idx3-ubyte",index+j,image);
				res = get_n_result("mnist_dataset/train-labels.idx1-ubyte",index+j);
				for(int l = 0; l <IN; l++)
				{
					image[l] = image[l]/100.0f;
				}
				aNet.run(image);
				result = aNet.results();
				int big=0;
				for(int k = 0; k < OUT;k++)
				{
					if(result[big] < result[k])
						big = k;
			//		cout << result[k] << " ";
				}
			//	cout << ": "<<res<<endl;
				if(res == big)
					numBigger++;
				aNet.reset();
			}
			cout << "Results: " << numBigger << " / " << TESTROUNDS << " Correct" << endl << endl;
		}
	}
	free(result);
	return 0;
}
