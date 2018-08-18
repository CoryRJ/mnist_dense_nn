#include "Cnn.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
Cnn::Cnn(int numOfLayers, int *layers)
{
	node *aNode;
	node *aNode2;
	node *spareN;
	weight *aWeight;
	weight *spareW;
	std::default_random_engine gen;
	std::normal_distribution<float> dis(0,1);
	layerDims = (int*)malloc(sizeof(int)*numOfLayers);
	for(int i = 0; i < numOfLayers; i++)
	{
		nodes.push_back( (struct node*)std::malloc(sizeof(struct node)));
		aNode = nodes[i];
		for(int j = 0; j <layers[i]; j++)
		{
			aNode->b=0;
			aNode->a=0;
			aNode->bias=0;
			aNode->bias_d=0;
			aNode->d=0;
			aNode->n =  (struct node*)std::malloc(sizeof(struct node));
			spareN = aNode;
			aNode = aNode->n;
		}
		spareN->n = NULL;
		std::free(aNode);
		layerDims[i] = layers[i];
	}
	//START: Complete graph nn
	for(int i = 0; i < numOfLayers-1; i++)
	{
		weights.push_back( (struct weight*)std::malloc(sizeof(struct weight)));
		aWeight = weights[i];
		aNode = nodes[i];
		aNode2 = nodes[i+1];
		for(int j = 0; j <layers[i]; j++)
		{
			for(int k = 0; k <layers[i+1]; k++)
			{
				aWeight->L = aNode;
				aWeight->R = aNode2;
				aWeight->w = dis(gen)*(sqrt(1.0/layerDims[i+1]));
//				std::cout << aWeight->w << std::endl;
				aWeight->d = 0; 
				aWeight->n =  (struct weight*)std::malloc(sizeof(struct weight));
				spareW = aWeight;
				aWeight = aWeight->n;
				aNode2 = aNode2->n;
			}
			aNode = aNode->n;
			aNode2 = nodes[i+1];
		}
		spareW->n = NULL;
		std::free(aWeight);
	}
	//END: Complete graph nn
	numLayers = numOfLayers;
}
float Cnn::error(float res, float tru)
{
	return 0.5*(res-tru)*(res-tru);
}
float Cnn::error_dir(float res,float tru)
{
	return res-tru;
}
float Cnn::softmax(float inp)
{
	return exp(inp)/total_soft;
}
float Cnn::softmax_dir(float inp)
{
	return (exp(inp)*(total_soft-exp(inp)))/(total_soft*total_soft);
}

void Cnn::backprop()
{
	node *aNode = nodes[numLayers-1];
	weight *aWeight;
	//START: right seen layer
//	std::cout << "soft_total: "<< total_soft << std::endl;
	while(aNode != NULL)
	{
//		std::cout << "R d er_d: "<< error_dir(softmax(aNode->b),0.5) << std::endl;
//		std::cout << "R d sf_d: "<< softmax_dir(aNode->b) << std::endl;
		aNode->d = error_dir(softmax(aNode->b),0.1)*softmax_dir(aNode->b);
		aNode->b = 0;
//		std::cout << "R d: "<< aNode->d << std::endl;
		aNode = aNode->n;
	}
	//END: right seen layer
	for(int i = numLayers-2; i > -1; i--)
	{
		aWeight = weights[i];
		while(aWeight != NULL)
		{
			aWeight->L->d += aWeight->R->d*aWeight->w;
//			std::cout << "b w: "<< aWeight->w << std::endl;
			aWeight->w -= learning_rate*aWeight->R->d*aWeight->L->a;
			if(isnan(aWeight->w))
			{
//				std::cout << "FOUND A NAN!" << std::endl;
			}
//			std::cout << "a w: "<< aWeight->w << std::endl;
			aWeight = aWeight->n;
		}
		aNode = nodes[i];
		while(aNode != NULL)
		{
			aNode->d = aNode->d*act_dir(aNode->b);
			aNode->b = 0;
			aNode = aNode->n;
		}
	}
}
void Cnn::setAct(float (*f)(float))
{
	act = f;
}
float Cnn::act_dir(float inp)
{
	if(inp > 0)
		return 1;
	return 0;
	float delta = 0.00001;
	return (act(inp+delta)-act(inp-delta))/(2*delta);
}

void Cnn::run(float *input)
{
	node *aNode = nodes[0];
	weight *aWeight;
	for(int i = 0; i < layerDims[0]; i++)
	{
		aNode->a = input[i];
		aNode->d = 0;
		aNode = aNode->n;
	}
	for(int i = 0; i < numLayers-1; i++)
	{
		aNode = nodes[i+1];
		aWeight = weights[i];
		while(aWeight != NULL)
		{
//			std::cout << "run L a: " << aWeight->L->a << std::endl;
//			std::cout << "run w: " << aWeight->w << std::endl;
			aWeight->R->b += aWeight->L->a*aWeight->w;
			aWeight = aWeight->n;
		}
		while(aNode != NULL)
		{
			aNode->a = act(aNode->b+aNode->bias);
			aNode->d = 0;
			aNode = aNode->n;
		}
	}
	aNode = nodes[numLayers-1];
	total_soft = 0;
	while(aNode != NULL)
	{
//		std::cout << "run soft b: " << aNode->b << std::endl;
//		std::cout << "run soft b: " << exp(aNode->b) << std::endl;
		total_soft += exp(aNode->b);
		aNode = aNode->n;
	}
}

float* Cnn::results()
{
	float* results = (float*) malloc(sizeof(float)*layerDims[numLayers-1]);
	node *aNode=nodes[numLayers-1];
	for(int i =0; i < layerDims[numLayers-1]; i++)
	{
		results[i] = softmax(aNode->b);
		aNode = aNode->n;
	}
	return results;
}
