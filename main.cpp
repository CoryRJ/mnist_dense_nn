#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include <random>
#include "TRI/to_read.h"
#include "nn_class/Cnn.h"
#define SIZE 5
#define PRE 4
#define OUT 10		//DO NOT CHANGE
#define IN 28*28	//DO NOT CHANGE
#define MID 28
#define TESTS 60000
#define TEST_EVERY 1000
#define TESTROUNDS 10000
#define FINAL_TEST 10000
#define BATCH 15
#define CORRECT 1.0f
using namespace std;

float tanH(float f)
{
	double v = (1.0-exp((double)(-2*f)))/(1.0+exp((double)(-2*f)));
	return (float)v;	
}
float tanH_d(float f)
{
	double v = 4*(exp((double)(f))/(1+exp((double)(2*f))))*(exp((double)(f))/(1+exp((double)(2*f))));
	return (float)v;	
}
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
void normalize_data(float v[IN])
{
	for(int i = 0; i < IN; i++)
	{
		v[i] = (v[i]*2-255)/255.0;
	}
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
	dims[1] = IN/14;
	dims[2] = IN/28;
	dims[3] = IN/56;
	dims[SIZE-1] = OUT;

	Cnn aNet(SIZE,dims);
//	aNet.setAct(sigmoid,sigmoid_d);
//	aNet.setAct(tanH,tanH_d);
	float *values = (float*) malloc(sizeof(float)*OUT);
	for(int i = 0; i < OUT; i++)
	{
		values[i] = (1-CORRECT)/9.0;
		wrong[i]=0;
		total[i]=0;
		result[i]=0;
	}
	int index = 0;
	int test_index = 0;
	int res = 0;
	int numBigger = 0;
	int big = 0;
	default_random_engine gen;
	uniform_int_distribution<int> dis(0,59999);
	uniform_int_distribution<int> dis2(0,9999);
	for(int i = 0; i < TESTS; i++)
	{
		if(i%TEST_EVERY == 0)
		{
			cout << "Starting test: " << endl;
			numBigger = 0;
			for(int j = 0; j < TESTROUNDS; j++)
			{
				
				test_index = dis2(gen);
				get_n("mnist_dataset/t10k-images.idx3-ubyte",test_index,image);
				res = get_n_result("mnist_dataset/t10k-labels.idx1-ubyte",test_index);
				normalize_data(image);
				aNet.run(image);
				aNet.results(result);
				int big=0;
				for(int k = 0; k < OUT;k++)
				{
					if(result[big] < result[k])
						big = k;
//					cout << result[k] << " ";
				}
//				cout << ": "<<res<<endl;
				if(res == big)
					numBigger++;
				aNet.reset();
//				test_index++;
			}

			cout << "Results: " << numBigger << " / " << TESTROUNDS << " Correct" << endl << endl;
		}
		aNet.reset();
		cout << "Starting batch: " << i+1 << " / " << TESTS << " ... ... ..." << endl;
		cout << "Batch size: " << BATCH << endl;
		for(int j = 0; j < BATCH; j++)
		{
			index = dis(gen);
			get_n("mnist_dataset/train-images.idx3-ubyte",index,image);
			normalize_data(image);
			res = get_n_result("mnist_dataset/train-labels.idx1-ubyte",index);
			values[res] = CORRECT;
			aNet.run(image);
			aNet.backprop(values);
			values[res] = (1-CORRECT)/9.0;
//			index++;
		}
		cout << "Done! Updating... ... ..." << endl;
		aNet.update();
	}
	numBigger = 0;
	cout << endl << "Starting final test: " << FINAL_TEST << " ... ... ..." << endl;
	for(int j = 0; j < FINAL_TEST; j++)
	{
		index = dis2(gen);
		get_n("mnist_dataset/t10k-images.idx3-ubyte",index,image);
		res = get_n_result("mnist_dataset/t10k-labels.idx1-ubyte",index);
		normalize_data(image);
		aNet.run(image);
		aNet.results(result);
		big=0;
		for(int k = 1; k < OUT;k++)
		{
			if(result[big] < result[k])
				big = k;
	//		cout << result[k] << " ";
		}
	//	cout << ": "<<res<<endl;
		if(res == big)
			numBigger++;
		else
			wrong[res]++;
		total[res]++;
		aNet.reset();
	}
	cout << "Results: " << numBigger << " / " << FINAL_TEST << " Correct" << endl << endl;
	cout << "Which it got wrong: " <<  endl;
	for(int i = 0; i < OUT;i++)
	{
		cout <<i<< ": " << wrong[i]<<" / " << total[i] << "  " << "Percent wrong: " << 100*(float)wrong[i]/(float)total[i] << endl;

	}
	free(result);
	return 0;
}
