#include "Cnn.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
Cnn::Cnn(int numOfLayers, int *layers)
{
	act = relu;
	act_dir = relu_dir;
	error = mse;
	error_dir = mse_dir;
	node *aNode;
	node *aNode2;
	node *spareN;
	weight *aWeight;
	weight *spareW;
	std::default_random_engine gen(121);
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
				aWeight->w = dis(gen)*(sqrt(1.0/layerDims[i]));
				aWeight->d_old=0;
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
float Cnn::softmax(float inp)
{
	double res = exp((double)inp)/total_soft;
//	if(isinf(res)||isnan(res))
//		return 0;
	return (float)res;
}
float Cnn::softmax_dir(float inp)
{
	double res = (exp((double)inp)/total_soft)*((total_soft-exp((double)inp))/total_soft);
//	if(isinf(res)||isnan(res))
//		return 1;
	return (float)res;
}

void Cnn::backprop(float *real_vals)
{
	batch++;
	node *aNode = nodes[numLayers-1];
	weight *aWeight;
	//START: right seen layer
//	std::cout << "soft_total: "<< total_soft << std::endl;
	int n = 0;
	while(aNode != NULL)
	{
		if(isnan(aNode->d))
		{
			std::cout << "nan1 aNode d: " << aNode->d << std::endl;
		}
		aNode->d = error_dir(softmax(aNode->b+aNode->bias),real_vals[n++],layerDims[numLayers-1])*softmax_dir(aNode->b+aNode->bias);
		if(isnan(aNode->d) || (aNode->d > 300)|| (aNode->d < -300))
		{
			std::cout << "nan2 aNode d: " << aNode->d << std::endl;
			std::cout << "nan2 er_d: " <<  error_dir(softmax(aNode->b),real_vals[n-1],layerDims[numLayers-1])<< std::endl;
			std::cout << "nan2 soft_max_d: " << softmax_dir(aNode->b) << std::endl;
			std::cout << "nan2 soft_max_d*er_d: " << error_dir(softmax(aNode->b),real_vals[n-1],layerDims[numLayers-1])*softmax_dir(aNode->b)<< std::endl;
			std::cout << "nan2 soft_max: " << softmax(aNode->b) << std::endl;
			std::cout << "nan2 aNode b: " << aNode->b << std::endl;
			std::cout << "nan2 exp(b): " << exp(aNode->b) << std::endl;
			std::cout << "nan2 soft_total: " << total_soft << std::endl;
			std::cout << "nan2 1.0/soft_total^2: " << 1.0/(total_soft*total_soft) << std::endl;
			std::cout << std::endl;
			aNode = nodes[numLayers-1];
			while(aNode != NULL)
			{
				std::cout << "all nan2 aNode b: " << aNode->b << std::endl;
				std::cout << "all nan2 aNode bias: " << aNode->bias << std::endl;
				aNode=aNode->n;
			}
			while(true){}
		}
		aNode->b = 0;
		aNode->bias_d += aNode->d;
		aNode = aNode->n;
	}
	//END: right seen layer
	for(int i = numLayers-2; i > -1; i--)
	{
		aWeight = weights[i];
		while(aWeight != NULL)
		{
	n++;
			aWeight->L->d += aWeight->R->d*aWeight->w;
			aWeight->d += aWeight->R->d*aWeight->L->a;
			if(isnan(aWeight->w) || isnan(aWeight->L->d) || (aWeight->d > 300)|| (aWeight->d < -300))
			{
				std::cout << "nan L d " << aWeight->L->d << std::endl;
				std::cout << "nan R d " << aWeight->R->d << std::endl;
				std::cout << "nan L a " << aWeight->L->a << std::endl;
				std::cout << "nan w " << aWeight->w << std::endl;
				std::cout << "nan w d " << aWeight->d << std::endl;
				std::cout << "layer " << i << std::endl;
				std::cout << "weight num " << n << std::endl;
				std::cout << "FOUND A NAN!" << std::endl;
				while(true){}
			}
			aWeight = aWeight->n;
		}
		aNode = nodes[i];
		while(aNode != NULL)
		{
			aNode->d = aNode->d*act_dir(aNode->b+aNode->bias);
			aNode->bias_d += aNode->d;
			aNode->b = 0;
			aNode = aNode->n;
		}
	}
	
}
void Cnn::setAct(float (*f)(float),float (*f_d)(float))
{
	act = f;
	act_dir = f_d;
}
void Cnn::setError(float (*f)(float,float,float),float (*f_d)(float,float,float))
{
	error = f;
	error_dir = f_d;
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
/*
		if(isnan(exp(aNode->b)) || isnan(total_soft) || isinf(1.0/(total_soft*total_soft)))
		{
			std::cout << "run soft b: " << aNode->b << std::endl;
			std::cout << "run soft b: " << exp(aNode->b) << std::endl;
			std::cout << "run soft_total: " << total_soft << std::endl;
			std::cout << "run 1.0/soft_total^2: " << 1.0f/(total_soft*total_soft) << std::endl;
//			while(true){}
		}*/
		total_soft += exp((double)(aNode->b+aNode->bias));
		aNode = aNode->n;
	}
}

void Cnn::results(float *result)
{
	node *aNode=nodes[numLayers-1];
	for(int i =0; i < layerDims[numLayers-1]; i++)
	{
//		std::cout <<"b: "<< aNode->b << " bias: " << aNode->bias << std::endl;
		result[i] = softmax(aNode->b+aNode->bias);
		aNode = aNode->n;
	}
//	std::cout << std::endl;
}

void Cnn::update()
{
	if(batch == 0)
		return;
	node *aNode = nodes[numLayers-1];
	weight *aWeight;
	int aNum = 0;
	for(int i = numLayers-2; i > -1; i--)
	{
		aNode = nodes[i+1];
		aWeight = weights[i];
		while(aWeight != NULL)
		{
			aWeight->d_old = momentum_fac*aWeight->d_old+learning_rate*aWeight->d/(float)batch;
			aWeight->w -= aWeight->d_old;
//			if(aNum < 20)
//				std::cout << "nan3 w d " << aWeight->d << std::endl;
			if(isnan(aWeight->w) || isnan(aWeight->L->d)||(aWeight->w >300))
			{
				std::cout << "nan3 L d " << aWeight->L->d << std::endl;
				std::cout << "nan3 R d " << aWeight->R->d << std::endl;
				std::cout << "nan3 L a " << aWeight->L->a << std::endl;
				std::cout << "nan3 w " << aWeight->w << std::endl;
				std::cout << "nan3 w d " << aWeight->d << std::endl;
				std::cout << "nan3 batch " << batch << std::endl;
				std::cout << "FOUND A NAN!" << std::endl;
				while(true){}
			}
			aNum++;
			aWeight->d = 0;
			aWeight = aWeight->n;
		}
		while(aNode != NULL)
		{
			aNode->bias_d_old = momentum_fac*aNode->bias_d_old + learning_rate*aNode->bias_d/(float)batch;
			aNode->bias -= aNode->bias_d_old;
			aNode->bias_d = 0;
			aNode = aNode->n;
		}
	}	
	batch = 0;
}

void Cnn::reset()
{
	
	node *aNode;
	weight *aWeight;
	for(int i = numLayers-2; i > -1; i--)
	{
		aWeight = weights[i];
		while(aWeight != NULL)
		{
			aWeight->d = 0.f;
			aWeight = aWeight->n;
		}
	}
	for(int i = numLayers-1; i > -1; i--)
	{
		aNode = nodes[i];
		while(aNode != NULL)
		{
			aNode->bias_d = 0.f;
			aNode->d = 0.f;
			aNode->b = 0.f;
			aNode->a = 0.f;
			aNode = aNode->n;
		}
	}
	total_soft = 0;
	batch = 0;
}

void Cnn::setLearnRate(float lr)
{
	learning_rate = lr;
}

float relu(float inp)
{
	if(inp > 0)
		return inp;
	return 0;
}
float relu_dir(float inp)
{
	if(inp > 0)
		return 1;
	return 0;
}
float mse(float res, float tru,float out)
{
	return 0.5*(res-tru)*(res-tru)/out;
}
float mse_dir(float res,float tru, float out)
{
	return (res-tru)/out;
}
