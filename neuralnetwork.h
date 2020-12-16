#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <math.h>
#include "array1d.h"
#include "matrixmn.h"

#include <QDebug>

#define MAX2(a, b) ((a) > (b) ? (a) : (b))

typedef double D;
typedef D (*ActivateFunction)(const D&);
typedef D (*GradientFunction)(const D&);
typedef D (*NNFunction)(const D&);

class NeuralNetwork
{
public:
    NeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers);

    bool set_nn_function_with_name(QString name);
    bool set_input_layer(const VectorND<D>& input);
    void set_learning_rate(const D& learning_rate) { learning_rate_ = learning_rate; }
    void update_weights(Array1D<MatrixMN<D>>& weights, Array1D<VectorND<D>>& grads, Array1D<VectorND<D>>& dSigma_dW);
    void print_output_layer() {
        std::cout << output_layer_ << std::endl;
    }

    void PropForward();
    void PropBackward(const VectorND<D>& y_target);

    static D getIdentity(const D& x);
    static D getIdentityGradFromY(const D& x);
    static D getRELU(const D& x);
    static D getRELUGradFromY(const D& x);//{} // RELU Grad from X == RELU Grad from Y

private:

    const int kRow = 0, kCol = 1;

    int num_inputs_, num_outputs_, num_hidden_layers_;

    int shape_hidden_layer_weight_[2];
    int shape_output_layer_weight_[2];

    D learning_rate_;

    NNFunction activate_function_;
    NNFunction gradient_function_;

    VectorND<D> input_layer_;
    VectorND<D> output_layer_;

    Array1D<MatrixMN<D> > weights_;

    Array1D<VectorND<D> > sigma_;
    Array1D<VectorND<D> > act_;

    Array1D<VectorND<D> > grads_;
    Array1D<VectorND<D> > dSigma_dW_;

    bool BuildNetwork();

    void RunNNFunction(NNFunction act_func, VectorND<D> &from, VectorND<D> &to);

    MatrixMN<D>& MultiplyElementByElement(VectorND<D>& vec, MatrixMN<D>& mat) {
        Q_ASSERT(vec.num_dimension_ == mat.num_rows_);

        for(int i=0;i<mat.num_cols_;i++) {
            for(int j=0;j<mat.num_rows_;j++) {
                mat.values_[mat.get1DIndex(j, i)] *= vec.values_[j];
            }
        }

        return mat;
    }
};

#endif // NEURALNETWORK_H
