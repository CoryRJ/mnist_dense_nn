#include <fstream>
#include <string>
#include <iostream>

using namespace std;

int charBitToInt(char a, int offset)
{
	unsigned int total = 0;
	for(int i = 0; i < 8; i++)
	{
		total += (unsigned int)(1 << (i+offset))*( (((1 << i)&a)>0?1:0));
	}
	return total;
}

//END:Function that reads in from a file
void get_n(string aFile,int n, int* out)
{
	ifstream file(aFile);
	string str; 

	getline(file, str);
	unsigned int width=n*28*28;
	bool notFound = true;
	if(width < str.length()-4*(4))
	{
		unsigned int offset = width+4*(4);
		for(int i = 0; i < 28*28; i++)
		{
			if(offset > str.length())
			{
				getline(file, str);
				offset = 0;
			}
			out[i] = charBitToInt(str[offset++],0);
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
				unsigned int offset = width;
				for(int i = 0; i < 28*28; i++)
				{
					if(offset > str.length())
					{
						getline(file, str);
						offset = 0;
					}
					out[i] = charBitToInt(str[offset++],0);
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
int get_n_result(string aFile,int n)
{
	ifstream file(aFile);
	string str; 

	getline(file, str);
	file.close();
	return charBitToInt(str[n+4*2],0);
}
