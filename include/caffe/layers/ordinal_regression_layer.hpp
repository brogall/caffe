#ifndef CAFFE_ORDINAL_REGRESSION_LAYER_HPP_
#define CAFFE_ORDINAL_REGRESSION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    

template <typename Dtype>
class OrdinalRegressionLayer : public Layer<Dtype> {
public:
    explicit OrdinalRegressionLayer(const LayerParameter& param)
    : Layer<Dtype>(param)/*, Xw(0), Df_w(0), Dtheta_2_Dgamma(0), Df_t(0), Dt_g(0), Df_x(0) */ {   }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "OrdinalRegression"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    void updateTheta_cpu();
    //void updateTheta_gpu();
    
    const Dtype* getTheta();
    
    const Dtype* getGamma2Theta();
    
    virtual ~OrdinalRegressionLayer();
    
protected:
    
#ifndef CPU_ONLY
    
    //void create_Dtheta_2_Dgamma(int n, float *A, const float *B);
    //void create_Dtheta_2_Dgamma(int n, double *A, const double *B);
    
#endif // CPU_ONLY
    
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    
#ifndef CPU_ONLY
    /*
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
     virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
     */
#endif
    
    virtual void printMatrix(const Dtype* ptr, int rows, int cols, bool top = false, bool bottom = false);
    
    
    /*
     Dtype* Xw;
     Dtype* Df_w;
     Dtype* Dtheta_2_Dgamma;
     Dtype* Df_t;
     Dtype* Dt_g;
     Dtype* Df_x;
     Dtype* tmpN;
     */
    
    Blob<Dtype> Xw;
    Blob<Dtype> Df_w;
    Blob<Dtype> Dtheta_2_Dgamma;
    Blob<Dtype> Df_t;
    Blob<Dtype> Df_g;
    Blob<Dtype> Df_x;
    Blob<Dtype> tmpN;
    
    
    int M_;
    int K_;
    int N_;
    int num_classes_;
    // this will store the lower triangular matrix that maps from gamma to theta
    Blob<Dtype> gamma2theta_;
    
    
    // variable to hold the theta interval vector (K classes), whereby theta_0 = -inf, theta_K = +inf
    Blob<Dtype> theta_;
    
    Blob<Dtype> map_matrix_;
    
    inline Dtype phi(Dtype val) { return 1.0/(1.+exp(-val));}
    inline Dtype Dphi(Dtype val) { return phi(val)*(1. - phi(val)); }
};

}  // namespace caffe

#endif