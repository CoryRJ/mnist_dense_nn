#ifndef CNN
#define CNN

class Cnn
{
	int numLayers;

public:
	Cnn(int numOfLayers);
	Cnn(int numOfLayers, int numOfNodes);
	int backprop();

private:
	int layerDims[];
	int nodes[];
	int weights[];

};


#endif  //CNN
