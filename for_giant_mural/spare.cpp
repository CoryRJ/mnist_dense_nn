#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

int charBitToInt(char a, int offset)
{
	unsigned int total = 0;
	for(int i = 0; i < 8; i++)
	{
		total += (unsigned int)(1 << i+offset)*( (((1 << i)&a)>0?1:0));
	}
	return total;
}

//http://stackoverflow.com/questions/13035674/how-to-read-line-by-line-or-a-whole-text-file-at-once
//My file format:
//If first character in line is a #, it is a comment
void readFile(string aFile)
{
	ifstream file(aFile);
	string str; 
	int dataType = -1;
	int dim = -1;
	int strLen;
	/*
	if(file.is_open())
		cout << "FILE IS OPEN" << endl;
	else
		cout << "FILE IS CLOSED" << endl;
	  */  
	getline(file, str);
	strLen = str.length();

	dataType = charBitToInt(str[2],8);
	dim = charBitToInt(str[3],0);
	int dims[dim];
	int dimsCalc[dim]={0};
	dimsCalc[0] = 2;
	
	int k = 4;
	for(int i = 0; i < dim; i++)
	{
		dims[i] = charBitToInt(str[k++],24) +
			charBitToInt(str[k++],16) +
			charBitToInt(str[k++],8) +
			charBitToInt(str[k++],0);
	}
	while(dimsCalc[0] < 3)
	{
		dimsCalc[dim-1] = 0;
		while(dimsCalc[dim-1] < dims[dim-1])
		{
			dimsCalc[dim-1]++;
			cout << charBitToInt(str[k++],0) << endl;
			if(k>strLen-1)
			{
				k = 0;
				getline(file, str);
				strLen = str.length();
			}
		}
		for(int j = dim-2;j > -1;j--)
		{
			if(dimsCalc[j] < dims[j]-1)
			{
				dimsCalc[j]++;
				j = -1;
			}
			else
			{
				dimsCalc[j] = 0;
			}
		}
//		cout << endl;
	}
	file.close();
}
//END:Function that reads in from a file
void get_n(string aFile,int n, int off)
{
	ifstream file(aFile);
	string str; 

	getline(file, str);
	int width=n*28*28 + 28*off;
	bool notFound = true;
	if(width < str.length()-4*(4))
	{
		int offset = width+4*(4);
		for(int i = 0; i < 28; i++)
		{
			if(offset > str.length())
			{
				getline(file, str);
				offset = 0;
			}
			cout << charBitToInt(str[offset++],0) << endl;
		}
		notFound = false;
	}
	else
	{
		width -= str.length() -4*(4);
		getline(file, str);
		while(notFound)
		{
			if(width < str.length())
			{
				int offset = width;
				for(int i = 0; i < 28; i++)
				{
					if(offset > str.length())
					{
						getline(file, str);
						offset = 0;
					}
					cout << charBitToInt(str[offset++],0) << endl;
				}
				notFound = false;
			}
			else
			{
				width -= str.length()+1;
				getline(file, str);
			}
		}
	}
	file.close();
}
//END:Function that reads in from a file
void get_n_result(string aFile,int n)
{
	ifstream file(aFile);
	string str; 

	getline(file, str);
	cout << charBitToInt(str[n+4*2],0) << endl;
	file.close();
}
//END:Function that reads in from a file
int main(int argc, char *argv[])
{	

	//readFile("mnist_dataset/aTest.txt");
	//readFile("mnist_dataset/train-labels.idx1-ubyte");
	//readFile("mnist_dataset/train-images.idx3-ubyte");
	for(int k = 0; k < 244; k++)
		for(int i =0; i < 28; i++)
		{
			for(int j = 244*k; j < 244*(k+1); j++)
				get_n("mnist_dataset/train-images.idx3-ubyte",j,i);
		}
	//get_n_result("mnist_dataset/train-labels.idx1-ubyte",49);
	return 0;
}
