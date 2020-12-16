#include "neuralnetwork.h"

using namespace std;

/*
const static float gk_test_weights_222[18] = {0.0123456, 0.0131538, 0.0755605,
                                              0.045865, 0.0532767, 0.0218959,

                                              0.0123456, 0.0131538, 0.0755605,
                                              0.045865, 0.0532767, 0.0218959,

                                              0.0123456, 0.0131538, 0.0755605,
                                              0.045865, 0.0532767, 0.0218959};

const static float gk_test_weights_211[9] = {0.0123456, 0.0131538, 0.0755605,
                                             0.045865, 0.0532767, 0.0218959,

                                             0.00470446, 0.0678865, 0.0679296};

const static float gk_test_weights_111[6] = {0.0123456, 0.0131538,

                                             0.0755605, 0.045865};

const static float gk_test_weights_121[6] = {0.0123456, 0.0131538,

                                             0.0755605, 0.045865,
                                             0.045865, 0.0532767};
*/


NeuralNetwork::NeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers)
    : num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      num_hidden_layers_(num_hidden_layers)
{
    if(!BuildNetwork()) {
        cout << "Network Build Failed!" << endl;
    }
}

void NeuralNetwork::RunNNFunction(NNFunction act_func, VectorND<D> &from, VectorND<D> &to) {

    Q_ASSERT(from.num_dimension_ == to.num_dimension_);
    Q_ASSERT(act_func != NULL);

    for(int i=0;i<from.num_dimension_;i++)
        to[i] = act_func(from[i]);

}

D NeuralNetwork::getIdentity(const D &x) {
    return x;
}

D NeuralNetwork::getIdentityGradFromY(const D &x) {
    return 1;
}

D NeuralNetwork::getRELU(const D &x) {
    return MAX2(x, 0.0);
}

D NeuralNetwork::getRELUGradFromY(const D &x) {
    return (x > 0.0 ? 1.0 : 0.0);
}

bool NeuralNetwork::BuildNetwork() {

    /*
     * VectorND<D> input_layer_, output_layer_ initialization
     */

    input_layer_.initialize(num_inputs_);
    output_layer_.initialize(num_outputs_);

    /*
     * Array1D<MatrixMN<D> > weights_ initialization
     */

    // + 1 is for output_layer_weight

    weights_.initialize(num_hidden_layers_ + 1);
    grads_.initialize(num_hidden_layers_ + 1);

    shape_hidden_layer_weight_[kRow] = num_inputs_;
    shape_hidden_layer_weight_[kCol] = num_inputs_ + 1;

    shape_output_layer_weight_[kRow] = num_outputs_;
    shape_output_layer_weight_[kCol] = num_inputs_ + 1;

    int num_hidden_layer_weight = shape_hidden_layer_weight_[kRow] * shape_hidden_layer_weight_[kCol];
    int num_output_layer_weight = shape_output_layer_weight_[kRow] * shape_output_layer_weight_[kCol];


    // for hidden layer weight
    int test_weight = 0;

    for(int i=0;i<weights_.num_elements_ - 1;i++) {
        weights_[i].initialize(shape_hidden_layer_weight_[kRow], shape_hidden_layer_weight_[kCol]);

        /* test
        for(int j=0;j<num_hidden_layer_weight;j++)
            weights_[i].values_[j] = gk_test_weights_211[test_weight++];
        */
        for(int j=0;j<num_hidden_layer_weight;j++)
            weights_[i].values_[j] = (D)rand() / RAND_MAX * 0.1;


    }

    // for output layer weight
    {
        int i = weights_.num_elements_ - 1;

        weights_[i].initialize(shape_output_layer_weight_[kRow], shape_output_layer_weight_[kCol]);

        /* test
        for(int j=0;j<num_output_layer_weight;j++)
            weights_[i].values_[j] = gk_test_weights_211[test_weight++];
        */

        for(int j=0;j<num_output_layer_weight;j++)
            weights_[i].values_[j] = (D)rand() / RAND_MAX * 0.1;
    }

    cout << "weights_.num_elements_ : " << weights_.num_elements_ << endl;


    for(int i=0;i<weights_.num_elements_;i++) {
        weights_[i].cout();
        cout << endl;
    }


    /*
     * Array1D<VectorND<D> > sigma_, act_ initialization
     */

    sigma_.initialize(num_hidden_layers_ + 1);
    act_.initialize(num_hidden_layers_ + 1);

    dSigma_dW_.initialize(num_hidden_layers_ + 1); // num of acts(-1 : last one) + input

    return true;
}


bool NeuralNetwork::set_nn_function_with_name(QString name) {
    Q_ASSERT(name.length() > 0);

    if(name == "relu"){
        activate_function_ = getRELU;
        gradient_function_ = getRELUGradFromY;
    }
    else
        return false;

    return true;
}

bool NeuralNetwork::set_input_layer(const VectorND<D> &input) {

    Q_ASSERT(input.num_dimension_ == num_inputs_);


    for(int i=0;i<input.num_dimension_;i++) {
        input_layer_[i] = input[i];

        cout << "input_layer_[" << i << "] : " << input_layer_[i] << endl;
    }

    return true;
}

void NeuralNetwork::update_weights(Array1D<MatrixMN<D> > &weights, Array1D<VectorND<D> > &grads, Array1D<VectorND<D> > &dSigma_dWs) {


    for(int i=0;i<weights.num_elements_;i++) {
        MatrixMN<D> dE_dW;
        dE_dW.initialize(weights_[i].num_rows_, weights_[i].num_cols_);

        VectorND<D> grad = grads[i];
        VectorND<D> dSigma_dW = dSigma_dWs[i];
        int num = 0;

        for(int r=0;r<dE_dW.num_rows_;r++) {
            for(int c=0;c<dE_dW.num_cols_;c++) {
                if(c == dE_dW.num_cols_ - 1) // for bias
                    dE_dW.values_[num++] = grad.values_[r] * 1;
                else
                    dE_dW.values_[num++] = grad.values_[r] * dSigma_dW.values_[c];
            }
        }

        MatrixMN<D> tmp = learning_rate_ * dE_dW;
        weights[i] += tmp;
    }
}

void NeuralNetwork::PropForward() {


    VectorND<D> x(num_inputs_ + 1);

    /*
     * PropForward in input layer
     */
    {

        x.copyPartial(input_layer_, 0, 0, num_inputs_);
        x[num_inputs_] = 1; // for bias

    }


    /*
     * PropForward in hidden layers
     */
    for(int i=0;i<num_hidden_layers_;i++) {
        sigma_[i].initialize(num_inputs_);
        act_[i].initialize(num_inputs_);

        weights_[i].multiply(x, sigma_[i]);

        RunNNFunction(activate_function_, sigma_[i], act_[i]);

        x.copyPartial(act_[i], 0, 0, num_inputs_);
        x[num_inputs_] = 1;
    }


    /*
     * PropForward in output layer
     */
    {

        int end = num_hidden_layers_;

        sigma_[end].initialize(num_outputs_);
        act_[end].initialize(num_outputs_);

        weights_[end].multiply(x, sigma_[end]);

        RunNNFunction(activate_function_, sigma_[end], act_[end]);

        output_layer_ = act_[end];

    }

}

void NeuralNetwork::PropBackward(const VectorND<D> &y_target) {

    /*
     * PropBackward in output layer
     *
     * grad = dE_dY * dY_dF * dF_dSigma
     *
     */


    VectorND<D> grad;
    int grads_index = grads_.num_elements_ - 1;
    {

        VectorND<D> dE_dY = y_target - output_layer_;
        VectorND<D> dY_dF(num_outputs_);
        RunNNFunction(getIdentityGradFromY, act_[act_.num_elements_ - 1], dY_dF);

        //dE_dY *= dY_dF;
        grad = dE_dY * dY_dF;

        VectorND<D> dF_dSigma(num_outputs_);
        RunNNFunction(getRELUGradFromY, sigma_[sigma_.num_elements_ - 1], dF_dSigma); // use gradient_function_ instead of getRELUGradFromY

        grad = grad * dF_dSigma;

        grads_[grads_index--]  = grad;
    }




    /*
     * PropBackward in hidden layer
     *
     * grad = grad * dSigma_dF * dF_dSigma * dSigma_dF * dF_dSigma ..... till the end
     *
     */
    {

        for(int i=0;i<num_hidden_layers_;i++) {

            MatrixMN<D> dSigma_dF = weights_[weights_.num_elements_ - 1 - i];

            cout << "show weights : " << endl;
            dSigma_dF.cout();

            dSigma_dF.delete_col(dSigma_dF.num_cols_ - 1); // delete bias elements

            dSigma_dF = MultiplyElementByElement(grad, dSigma_dF);

            VectorND<D> dF_dSigma(num_inputs_);
            RunNNFunction(getRELUGradFromY, sigma_[sigma_.num_elements_ - 2 - i], dF_dSigma);

            grad.initialize(num_inputs_);


            // test
            /*
            cout << "dSigma_dF : " << endl;
            dSigma_dF.cout();

            cout << "dF_Sigma : " << dF_dSigma << endl;
            */

            dSigma_dF.multiplyTransposed(dF_dSigma, grad);

//            cout << "grad : " << grad << endl;


            grads_[grads_index--] = grad;

        }

    }


    /*
     * dSigma_dW_ for PropBackward
     */
    dSigma_dW_[0] = input_layer_;
    for(int i=1;i<dSigma_dW_.num_elements_;i++) {
        dSigma_dW_[i] = act_[i-1];
    }

    update_weights(weights_, grads_, dSigma_dW_);

}
