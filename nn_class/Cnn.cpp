#include "Cnn.h"
#include <cstdlib>
#include <iostream>
Cnn::Cnn(int numOfLayers, int *layers)
{
	node *aNode;
	for(int i = 0; i < numOfLayers; i++)
	{
		nodes.push_back( (struct node*)std::malloc(sizeof(struct node)));
		aNode = nodes[i];
		for(int j = 0; j <layers[i]; j++)
		{
			aNode->n =  (struct node*)std::malloc(sizeof(struct node));
			aNode = aNode->n;
		}
		std::cout << layers[i] << std::endl;
	}
	
	numLayers = numOfLayers;
}
float Cnn::error(float inp){return inp;}
float Cnn::error_dir(float inp){return inp;}
float Cnn::softmax_dir(float inp){return inp;}
float Cnn::softmax(float inp){return inp;}

int Cnn::backprop()
{
	int i = numLayers -1;
	//START: right seen layer
	while(nodes[i]->n != NULL)
	{
		nodes[i]->d = error_dir(softmax((nodes[i])->a))*softmax_dir(nodes[i]->a);
	}
	//END: right seen layer
	return numLayers;
	
}
void Cnn::setAct(float (*f)(float))
{
	act = f;
}
float Cnn::act_dir(float inp)
{
	float delta = 0.00001;
	return (act(inp+delta)-act(inp-delta))/(2*delta);
}
