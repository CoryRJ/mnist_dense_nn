#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "TRI/to_read.h"
#include "nn_class/Cnn.h"
#define SIZE 5
#define PRE 4
#define OUT 10
#define IN 28*28
#define MID 28*28
#define TESTS 600
#define TEST_EVERY 10
#define TESTROUNDS 100
#define FINAL_TEST 10000
#define BATCH 50
using namespace std;

float sigmoid(float f)
{
	float v = exp(f)/(1+exp(f));
	if(isnan(v))
	{
		if(signbit(f))
			return 0.0f;
		return 1.0f;
	}
	return v;	
}
float sigmoid_d(float f)
{
	float v = exp(f)/(1+exp(f)*(1+exp(f)));
	if(isnan(v))
	{
		if(signbit(f))
			return 0.0f;
		return 1.0f;
	}
	return v;	
}

int main(int argc, char *argv[])
{	

	cout << setprecision(PRE);
	float image[IN];
	int dims[SIZE];
	float *result = (float*)malloc(sizeof(float)*OUT);
	float *wrong = (float*)malloc(sizeof(float)*OUT);
	float *total = (float*)malloc(sizeof(float)*OUT);
	for(int i = 0; i < SIZE; i++)
	{
		dims[i] = MID;
	}
	dims[0] = IN;
	dims[SIZE-1] = OUT;

	Cnn aNet(SIZE,dims);
//	aNet.setAct(sigmoid,sigmoid_d);
	float *values = (float*) malloc(sizeof(float)*10);
	for(int i = 0; i < OUT; i++)
	{
		values[i] = 0;
		wrong[i]=0;
		total[i]=0;
		result[i]=0;
	}
	int index = 0;
	int test_index = 0;
	int res = 0;
	int numBigger = 0;
	for(int i = 0; i < TESTS; i++)
	{
		aNet.reset();
		cout << "Starting batch: " << i << " / " << TESTS << " ... ... ..." << endl;
		cout << "Batch size: " << BATCH << endl;
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
		if(i%TEST_EVERY == 0)
		{
			numBigger = 0;
			cout << "Starting test: " << TESTROUNDS << " ... ... ..." << endl;
			for(int j = 0; j < TESTROUNDS; j++)
			{
				
				get_n("mnist_dataset/t10k-images.idx3-ubyte",test_index,image);
				res = get_n_result("mnist_dataset/t10k-labels.idx1-ubyte",test_index);
				for(int l = 0; l <IN; l++)
				{
					image[l] = image[l]/100.0f;
				}
				aNet.run(image);
				aNet.results(result);
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
				test_index++;
			}
			cout << "Results: " << numBigger << " / " << TESTROUNDS << " Correct" << endl << endl;
		}
		if(test_index >9999)
			test_index = 0;
		if(index >59999)
			index = 0;
	}
	numBigger = 0;
	cout << endl << "Starting final test: " << FINAL_TEST << " ... ... ..." << endl;
	for(int j = 0; j < FINAL_TEST; j++)
	{
		
		get_n("mnist_dataset/t10k-images.idx3-ubyte",j,image);
		res = get_n_result("mnist_dataset/t10k-labels.idx1-ubyte",j);
		for(int l = 0; l <IN; l++)
		{
			image[l] = image[l]/100.0f;
		}
		aNet.run(image);
		aNet.results(result);
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
		else
			wrong[big]++;
		total[res]++;
		aNet.reset();
	}
	cout << "Results: " << numBigger << " / " << FINAL_TEST << " Correct" << endl << endl;
	cout << "Which it got wrong: " <<  endl;
	for(int i = 0; i < OUT;i++)
	{
		cout <<i<< ": " << wrong[i] << ". " << "Percent wrong: " << 100*(float)wrong[i]/(float)total[i] << endl;

	}
	free(result);
	return 0;
}
