#ifndef CNN
#define CNN
#include <vector>
class Cnn
{
public:
	int numLayers;
	float (*error)(float, float,float);
	float (*error_dir)(float,float,float);
	float (*act)(float);
	float (*act_dir)(float);
	float softmax(float inp);
	float softmax_dir(float inp);
	float learning_rate = .1;
	float total_soft = 0;
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
	void backprop(float *real_vals);
	void update();
	void reset();
	void setAct(float (*f)(float),float (*f_d)(float));
	void setError(float (*f)(float,float,float),float (*f_d)(float,float,float));
	void setLearnRate(float lr);
	void run(float *input);
	void results(float *results);
	void destroy();

private:
	std::vector<node*> nodes;
	std::vector<weight*> weights;
	int *layerDims;

};
float relu(float inp);
float relu_dir(float inp);
float mse(float res, float tru,float out);
float mse_dir(float res,float tru,float out);
#endif  //CNN
