#ifndef CNN
#define CNN
#include <vector>
class Cnn
{
public:
	int numLayers;
	float error(float res, float tru);
	float error_dir(float res,float tru);
	float act_dir(float inp);
	float softmax(float inp);
	float softmax_dir(float inp);
	float total_soft = 0;
	float (*act)(float);
	void backprop(float *real_vals);
	void update();
	void reset();
	float learning_rate = .01;
	float sum = 0;
	int batch = 0;
	struct node
	{
		float b=0;
		float a=0;
		float bias=0;
		float bias_d=0;
		float d=0;
		node *n = NULL;
	};
	struct weight
	{
		node *L=NULL;
		node *R=NULL;
		float w=0;
		float d=0;
		weight *n = NULL;
		weight *alt = NULL; //if NULL, is original wait. Otherwise, update the thing in the address
	};

public:
	Cnn(int numOfLayers, int *layers);
	Cnn(int numOfLayers, int numOfNodes);
	void setAct(float (*f)(float));
	void setErr(float (*f)(float));
	void run(float *input);
	float* results();

private:
	std::vector<node*> nodes;
	std::vector<weight*> weights;
	int *layerDims;

};


#endif  //CNN
