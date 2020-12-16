#include <iostream>

#include "neuralnetwork.h"
#include "VectorND.h"

using namespace std;

int main()
{
    const int kNumInput = 5; const int kNumOutput = 3;
    const int kNumHiddenLayers = 2;

    const int kTrainingCount = 500;

    const D kLearningRate = 0.01;

    // Create Input Vector
    VectorND<D> x(kNumInput);
    x[0] = 0.0; x[1] = 1.5;
    x[2] = 3.0; x[3] = 0.0;
    x[4] = -10.0;

    // Create Target Vector
    VectorND<D> y_target(kNumOutput);
    y_target[0] = 2.0;
    y_target[1] = 1.0;
    y_target[2] = 10.0;

    // Create N.N
    NeuralNetwork nn(kNumInput, kNumOutput, kNumHiddenLayers);

    nn.set_input_layer(x);
    nn.set_learning_rate(kLearningRate);
    nn.set_nn_function_with_name("relu");

    for(int i=0;i<kTrainingCount;i++) {
        cout << "Count " << i << endl;
        cout << "y_target : " << y_target << endl;
        cout << "y : ";

        nn.print_output_layer();

        nn.PropForward();
        nn.PropBackward(y_target);
    }

    return 0;
}
