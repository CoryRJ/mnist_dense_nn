#include <iostream>
#include <fstream>
#include <string>
#include "TRI/to_read.h"

using namespace std;

struct node
{
	node* left;
	node* right;
	float w = 0;
	float g = 0;
	int n = 0;
};

int main(int argc, char *argv[])
{	

	int image[784];
	get_n("mnist_dataset/train-images.idx3-ubyte",stoi(argv[1]),image);
	for(int i = 0; i < 28; i++)
	{
		for(int j = 0; j < 28; j++)
			cout << image[28*i +j] << " ";
		cout << endl;
	}
	cout << get_n_result("mnist_dataset/train-labels.idx1-ubyte",stoi(argv[1])) << endl;
	return 0;
}
