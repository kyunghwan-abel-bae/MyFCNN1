#ifndef MATRIXMN_H
#define MATRIXMN_H

#include "vectornd.h"

#define SAFE_DELETE_ARRAY(pointer) if(pointer != nullptr){delete [] pointer; pointer=nullptr;}


template<class T>
class MatrixMN
{
public:
    int num_rows_;  // m_
    int num_cols_;  // n_
    T   *values_;

    MatrixMN()
        : values_(nullptr), num_rows_(0), num_cols_(0)
    {}

    ~MatrixMN() {
        SAFE_DELETE_ARRAY(values_);
    }

    void initialize(const int& _m, const int& _n, const bool init = true);

    void multiply(const VectorND<T>& vector, VectorND<T>& result) const;
    void multiplyTransposed(const VectorND<T>& vector, VectorND<T>& result) const;

    int get1DIndex(const int& row, const int& column) const;
    T&  getValue(const int& row, const int& column) const;

    void cout();

    // added by KH
    MatrixMN(const MatrixMN<T>& from) {
        num_rows_ = from.num_rows_;
        num_cols_ = from.num_cols_;

        values_ = new T[num_rows_ * num_cols_];

        for(int i=0;i<num_rows_*num_cols_;i++) {
            values_[i] = from.values_[i];
        }
    }

    void operator = (const MatrixMN<T>& from) {
        num_rows_ = from.num_rows_;
        num_cols_ = from.num_cols_;

        SAFE_DELETE_ARRAY(values_);

        values_ = new T[num_rows_ * num_cols_];

        for(int i=0;i<num_rows_*num_cols_;i++) {
            values_[i] = from.values_[i];
        }
    }


    void operator += (const MatrixMN<T>& from) {

        assert(num_rows_ == from.num_rows_);
        assert(num_cols_ == from.num_cols_);

        for(int i=0;i<num_rows_*num_cols_;i++) {
            values_[i] += from.values_[i];
        }
    }

    friend MatrixMN<T> operator * (const T& num, const MatrixMN<T>& from) {

        MatrixMN<T> ret;

        ret.initialize(from.num_rows_, from.num_cols_);

        for(int i=0;i<ret.num_rows_*ret.num_cols_;i++)
            ret.values_[i] = num * from.values_[i];

        return ret;
    }

    void delete_row(const int& row_index);

    void delete_col(const int& col_index);
};


#endif // MATRIXMN_H
