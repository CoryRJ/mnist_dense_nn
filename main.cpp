#include <iostream>
#include <fstream>
#include <string>
#include "TRI/to_read.h"
#include "nn_class/Cnn.h"
#define SIZE 5
using namespace std;

float x_s(float f)
{
	return f*f;
}

int main(int argc, char *argv[])
{	
/*
	int image[784];
	get_n("mnist_dataset/train-images.idx3-ubyte",stoi(argv[1]),image);

	for(int i = 0; i < 28; i++)
	{
		for(int j = 0; j < 28; j++)
			cout << image[28*i +j] << " ";
		cout << endl;
	}

	cout << get_n_result("mnist_dataset/train-labels.idx1-ubyte",stoi(argv[1])) << endl;

*/
	
	int dims[SIZE];
	for(int i = 0; i < SIZE; i++)
	{
		dims[i] = 24*24;
	}
	Cnn aNet(SIZE,dims);
	aNet.setAct(&x_s);
	cout << aNet.act(28.2) << endl;
	cout << aNet.act_dir(28.2) << endl;
	return 0;
}
