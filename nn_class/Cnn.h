#ifndef CNN
#define CNN
#include <vector>
class Cnn
{
public:
	int numLayers;
	float error(float inp);
	float error_dir(float inp);
	float act_dir(float inp);
	float softmax(float inp);
	float softmax_dir(float inp);
	float (*act)(float);
	int backprop();
	struct node
	{
		float b;
		float a;
		float d;
		node *n = NULL;
	};
	struct weight
	{
		node *L=NULL;
		node *R=NULL;
		float w;
		float d;
		weight *n = NULL;
		weight *alt = NULL; //if NULL, is original wait. Otherwise, update the thing in the address
	};

public:
	Cnn(int numOfLayers, int *layers);
	Cnn(int numOfLayers, int numOfNodes);
	void setAct(float (*f)(float));
	void setErr(float (*f)(float));

private:
	std::vector<node*> nodes;
	std::vector<weight*> weights;
	int layerDims[];

};


#endif  //CNN
