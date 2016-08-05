#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>


#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/ordinal_regression_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::printMatrix(const Dtype* ptr, int rows, int cols, bool top, bool bottom) {


        if(top)
            std::cout <<"----------------------------------------------------------" << std::endl;
        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {

                if ( c < cols-1)
                    std::cout << ptr[r*cols+c] << ", ";
                else
                    std::cout << ptr[r*cols+c];
            }
            std::cout << std::endl;
        }

        if(bottom)
            std::cout << "==========================================================" << std::endl;


    }

    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::LayerSetUp(
                                                           const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {



        Layer<Dtype>::LayerSetUp(bottom, top);

        OrdinalRegressionParameter ordinal_regression_param = this->layer_param_.ordinal_regression_param();

        //top.resize(2);

        CHECK( ordinal_regression_param.has_classes() ) << "Number of classes needs to be specified";


        const int axis = bottom[0]->CanonicalAxisIndex(1);
        // Dimensions starting from "axis" are "flattened" into a single
        // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
        // and axis == 1, N inner products with dimension CHW are performed.
        K_ = bottom[0]->count(axis);


        this->num_classes_ = ordinal_regression_param.classes();

        this->N_ = this->num_classes_-1;
        this->M_ = this->N_-1;

        int dim = bottom[0]->count() / bottom[0]->shape(0);


        if (this->blobs_.size() > 0)
        {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            LOG(INFO) << "Creating random blobs";
            this->blobs_.resize(2); // weight, theta


            // Intialize the weight
            vector<int> weight_shape(1);
            weight_shape[0] = dim;
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            // fill the weights
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                                                                      this->layer_param_.ordinal_regression_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());

            vector<int> gamma_shape(1);
            gamma_shape[0] = N_;
            this->blobs_[1].reset(new Blob<Dtype>(gamma_shape));
            shared_ptr<Filler<Dtype> > gamma_filler(GetFiller<Dtype>(
                                                                     this->layer_param_.ordinal_regression_param().gamma_filler()));
            gamma_filler->Fill(this->blobs_[1].get());
        }


        vector<int> map_shape(2);
        map_shape[0] = this->N_;
        map_shape[1] = this->M_;
        this->gamma2theta_.Reshape(map_shape);

        // initialize lower triangular matrix that maps gamma to theta representation

        Dtype* triangularData = this->gamma2theta_.mutable_cpu_data();


        for (int _row=0;_row<this->N_;_row++)
        {

            for (int _col=0; _col<this->M_; _col++)
            {

                if ( _row > _col) // 1
                    *triangularData++ = (Dtype)1.0;
                else // 0
                    *triangularData++ = (Dtype)0.0;

            }
        }


        vector<int> theta_vector_shape(1);
        theta_vector_shape[0] = this->num_classes_+1;
        this->theta_.Reshape(theta_vector_shape);

        Dtype* theta = this->theta_.mutable_cpu_data();

        // set the boundary values
        theta[0] = -std::numeric_limits<Dtype>::max()+10;
        theta[this->num_classes_] = std::numeric_limits<Dtype>::max()-10;


        this->param_propagate_down_.resize(this->blobs_.size(), true);

         {
             int num = bottom[0]->shape(0);
             int dim = bottom[0]->count() / bottom[0]->shape(0);


           /*
            Xw = new Dtype[num];

            Df_w = new Dtype[this->num_classes_*dim];

            Dtheta_2_Dgamma = new Dtype[(this->num_classes_+1)*(this->num_classes_-1)];

            Df_t = new Dtype[(this->num_classes_+1)*(this->num_classes_)];

            Dt_g = new Dtype[(this->num_classes_)*(this->num_classes_-1)];

            Df_x = new Dtype[this->num_classes_*dim];

            Dtype* tmpN = new Dtype[this->N_];

            */
            {
                vector<int> Xw_shape(1);
                Xw_shape[0] = num;
                this->Xw.Reshape(Xw_shape);
            }

            {
                vector<int> Dfw_shape(2);
                Dfw_shape[0] = this->num_classes_;
                Dfw_shape[1] = dim;
                this->Df_w.Reshape(Dfw_shape);
            }

            {
                vector<int> Dtheta2Dgamma_shape(2);
                Dtheta2Dgamma_shape[0] = (this->num_classes_+1);
                Dtheta2Dgamma_shape[1] = (this->num_classes_-1);
                this->Dtheta_2_Dgamma.Reshape(Dtheta2Dgamma_shape);
            }

            {
                vector<int> Dft_shape(2);
                Dft_shape[0] = (this->num_classes_+1);
                Dft_shape[1] = (this->num_classes_);
                this->Df_t.Reshape(Dft_shape);
            }

            {
                vector<int> Dfg_shape(2);
                Dfg_shape[0] = (this->num_classes_);
                Dfg_shape[1] = (this->num_classes_-1);
                this->Df_g.Reshape(Dfg_shape);
            }

            {
                vector<int> Dfx_shape(2);
                Dfx_shape[0] = (this->num_classes_);
                Dfx_shape[1] = dim;
                this->Df_x.Reshape(Dfx_shape);
            }

             {
                vector<int> tmpN_shape(1);
                tmpN_shape[0] = this->N_;
                this->tmpN.Reshape(tmpN_shape);
            }
        }

    }

    template <typename Dtype>
    OrdinalRegressionLayer<Dtype>::~OrdinalRegressionLayer()
    {
        /*if ( Xw) {
            delete[] Xw;
        }

        if (Df_w) {
            delete[] Df_w;
        }

        if (Dtheta_2_Dgamma) {
            delete[] Dtheta_2_Dgamma;
        }

        if (Df_t) {
            delete[] Df_t;
        }

        if (Dt_g) {
            delete[] Dt_g;
        }

        if (Df_x) {
            delete[] Df_x;
        }

        if (tmpN)
        {
            delete[] tmpN;
        }

        */

    }

    template <typename Dtype>
    const Dtype* OrdinalRegressionLayer<Dtype>::getTheta()
    {
        return this->theta_.cpu_data();
    }

    template <typename Dtype>
    const Dtype* OrdinalRegressionLayer<Dtype>::getGamma2Theta()
    {
        return this->gamma2theta_.cpu_data();
    }



    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::updateTheta_cpu()
    {

        caffe_copy(this->N_, this->blobs_[1]->cpu_data(), tmpN.mutable_cpu_data());

        // square the elements except for the first (which can be negative)
        caffe_powx(this->M_, (Dtype*)&tmpN.cpu_data()[1], Dtype(2), &tmpN.mutable_cpu_data()[1]);
        // matrix * vector
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->N_, this->M_, (Dtype)1., this->gamma2theta_.cpu_data(),
                              &tmpN.cpu_data()[1], 0.,
                              &this->theta_.mutable_cpu_data()[1]);

        caffe_add_scalar(this->N_, tmpN.cpu_data()[0], &this->theta_.mutable_cpu_data()[1] );

    }

    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::Reshape(
                                                        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


        int num = bottom[0]->shape(0);
        int dim = bottom[0]->count() / bottom[0]->shape(0);

        OrdinalRegressionParameter ordinal_regression_param = this->layer_param_.ordinal_regression_param();

        vector<int> top_shape1(2);

        top_shape1[0] = num;
        top_shape1[1] = ordinal_regression_param.classes();

        // Probabilities for each element and class
        top[0]->Reshape(top_shape1);

        vector<int> blob1_shape(1);
        blob1_shape[0] = dim;

        this->blobs_[0]->Reshape(blob1_shape);

        vector<int> blob2_shape(1);
        blob2_shape[0] = ordinal_regression_param.classes()-1;

        this->blobs_[1]->Reshape(blob2_shape);

        /*if ( Xw) {
            delete[] Xw;
            Xw = new Dtype[num];
        }

        if (Df_w) {
            delete[] Df_w;
            Df_w = new Dtype[this->num_classes_*dim];
        }

        if (Dtheta_2_Dgamma) {
            delete[] Dtheta_2_Dgamma;
            Dtheta_2_Dgamma = new Dtype[(this->num_classes_+1)*(this->num_classes_-1)];
        }

        if (Df_t) {
            delete[] Df_t;
            Df_t = new Dtype[(this->num_classes_+1)*(this->num_classes_)];
        }

        if (Dt_g) {
            delete[] Dt_g;
            Dt_g = new Dtype[(this->num_classes_)*(this->num_classes_-1)];
        }

        if (Df_x) {
            delete[] Df_x;
            Df_x = new Dtype[this->num_classes_*dim];
        }

        if (tmpN) {
            delete[] tmpN;
            tmpN = new Dtype[this->N_];
        }
        */

        {
                vector<int> Xw_shape(1);
                Xw_shape[0] = num;
                this->Xw.Reshape(Xw_shape);
            }

            {
                vector<int> Dfw_shape(2);
                Dfw_shape[0] = this->num_classes_;
                Dfw_shape[1] = dim;
                this->Df_w.Reshape(Dfw_shape);
            }

            {
                vector<int> Dtheta2Dgamma_shape(2);
                Dtheta2Dgamma_shape[0] = (this->num_classes_+1);
                Dtheta2Dgamma_shape[1] = (this->num_classes_-1);
                this->Dtheta_2_Dgamma.Reshape(Dtheta2Dgamma_shape);
            }

            {
                vector<int> Dft_shape(2);
                Dft_shape[0] = (this->num_classes_+1);
                Dft_shape[1] = (this->num_classes_);
                this->Df_t.Reshape(Dft_shape);
            }

            {
                vector<int> Dfg_shape(2);
                Dfg_shape[0] = (this->num_classes_);
                Dfg_shape[1] = (this->num_classes_-1);
                this->Df_g.Reshape(Dfg_shape);
            }

            {
                vector<int> Dfx_shape(2);
                Dfx_shape[0] = (this->num_classes_);
                Dfx_shape[1] = dim;
                this->Df_x.Reshape(Dfx_shape);
            }

             {
                vector<int> tmpN_shape(1);
                tmpN_shape[0] = this->N_;
                this->tmpN.Reshape(tmpN_shape);
            }

    }

    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::Forward_cpu(
                                                            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


        updateTheta_cpu();



        const Dtype* bottom_data = bottom[0]->cpu_data();
        int num = bottom[0]->shape(0);
        int dim = bottom[0]->count() / bottom[0]->shape(0);

        //Dtype* Xw = new Dtype[num];

        // X * w = Xw (num)

        // X SZ: (num) x (dim)
        // w SZ: (dim)

        caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_data,
                              this->blobs_[0]->cpu_data(), 0,
                              Xw.mutable_cpu_data());


        Dtype* prob_ = top[0]->mutable_cpu_data();
        for (int i = 0; i < num; ++i) {
            Dtype prev = 0.0;
            for (int label=1; label<=this->num_classes_ ; ++label) {
                // range 1..K
                Dtype z_ = (Xw.cpu_data()[i]-this->theta_.cpu_data()[label]);
                Dtype tmp =  phi(-z_);
                *prob_++ = (tmp-prev);
                prev = tmp;
            }
        }

    }


    template <typename Dtype>
    void OrdinalRegressionLayer<Dtype>::Backward_cpu(
                                                             const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,

                                                             const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[0]) {

            updateTheta_cpu();


            const Dtype* bottom_data = bottom[0]->cpu_data();

            int num = bottom[0]->shape(0);
            int dim = bottom[0]->count() / bottom[0]->shape(0);


            // Operation: X * w => (num)

            // X SZ: (num) x (dim)
            // w SZ: (dim)

            caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, (Dtype)1., bottom_data,
                                  this->blobs_[0]->cpu_data(), 0,
                                  Xw.mutable_cpu_data());


            // gradient w.r.t. weight
            if (this->param_propagate_down_[0]) {



                // Differential matrix: Df / Dw : [num rows (batch size),  dim cols (dim of data) )]

                caffe_memset(dim * sizeof(Dtype), 0, this->blobs_[0]->mutable_cpu_diff());

                for (int i = 0; i < num; ++i) {

                    //memset(Df_w, 0, this->num_classes_*dim*sizeof(Dtype));
                    caffe_memset( this->num_classes_*dim*sizeof(Dtype), 0, Df_w.mutable_cpu_data() );
                    for (int label=1; label<=this->num_classes_; ++label) {


                        // (phi(x) - phi(y))' => Dphi(x)-Dphi(y)
                        // equivalent to (and more numerically stable):
                        // (1 - phi(x) - phi(y))

                        Dtype scalar = (Dphi(this->theta_.cpu_data()[label] - Xw.cpu_data()[i]) - Dphi(this->theta_.cpu_data()[label-1] - Xw.cpu_data()[i]));

                        // scalar * X + Df_w -> Df_w
                        // bottom_data SZ: (num) x (dim)
                        // Df_w SZ: (this->num_classes_) x (dim)
                        caffe_axpy(dim, -scalar, &bottom_data[i*dim], &Df_w.mutable_cpu_data()[(label-1)*dim]);


                    }

                    // add: matrix * vector (transposed!)
                    // Operation:  for m in range(num) Df_w *  delta[:,m]
                    // Df_w SZ: (this->num_classes_) x (dim) => (dim)
                    // top[0]->cpu_diff() SZ: (num) x ( this->num_classes)
                    // delta SZ: ( this->num_classes)
                    caffe_cpu_gemv<Dtype>(CblasTrans, this->num_classes_, dim, (Dtype)1., Df_w.cpu_data(),
                                          &top[0]->cpu_diff()[i*this->num_classes_], (Dtype)1.,
                                          this->blobs_[0]->mutable_cpu_diff());
                }



                //delete[] Df_w;


            }

            // gradient w.r.t. gamma

            if (this->param_propagate_down_[1]) {



                // Differential matrix: Df / Dtheta : [ num_classes rows x num_classes+1 cols ]


                caffe_memset((this->num_classes_-1) * sizeof(Dtype), 0, this->blobs_[1]->mutable_cpu_diff());



                //memset(Dtheta_2_Dgamma, 0, (this->num_classes_+1)*(this->num_classes_-1)*sizeof(Dtype));
                caffe_memset((this->num_classes_+1)*(this->num_classes_-1)*sizeof(Dtype), 0, Dtheta_2_Dgamma.mutable_cpu_data() );

                // first row,last are zero ( not mapped to any gamma)
                Dtype *ptr = &Dtheta_2_Dgamma.mutable_cpu_data()[this->num_classes_-1];
                // we ignore theta_0 and theta_K!!

                for (int _row=0;_row<this->num_classes_-1;++_row) // leave last row zero

                {
                    for (int _col=0; _col <this->num_classes_-1;++_col)
                    {
                        if ( _row >= _col) // 1
                        {
                            if ( _col == 0 )
                                *ptr++ = (Dtype)1.0;
                            else {
                                *ptr++ = (Dtype)2.0* this->blobs_[1]->cpu_data()[_col]; // plug in the gamma values

                            }
                        }
                        else  { //  0
                            //*ptr ++ = (Dtype)0.0;
                            break; }
                    }
                }




                for (int i = 0; i < num; ++i) {
                    //memset(Df_t, 0, this->num_classes_*(this->num_classes_+1)*sizeof(Dtype));
                    caffe_memset( this->num_classes_*(this->num_classes_+1)*sizeof(Dtype), 0, Df_t.mutable_cpu_data() );

                    // from 1 .. K-1 (because of ignoring theta_0, theta_K
                    int pos = 1+(this->num_classes_ + 1);     // (this->num_classes + 1) because this matrix has (this->num_classes + 1) columns
                    for (int label=2; label<this->num_classes_; ++label) {

                        //  (phi(x))' => (1 - phi(x)) * phi(x)'

                        Dtype scalar_class_n = Dphi(this->theta_.cpu_data()[label] - Xw.cpu_data()[i]);
                        Dtype scalar_class_n_1 = Dphi(this->theta_.cpu_data()[label-1] - Xw.cpu_data()[i]);


                        Df_t.mutable_cpu_data()[pos]  = -scalar_class_n_1 ;
                        Df_t.mutable_cpu_data()[pos+1] = scalar_class_n ;




                        // Df_t SZ: (this->num_classes_) x (this->num_classes+1)
                        // Df_t has the following shape
                        // | -Dphi(label_0  ..) 0 .......    0 |
                        // | 0 -Dphi(label_0 - ..) Dphi(label_1 - ..) 0 ....0 |
                        // | 0     .........0 -Dphi(label_N-1 - ...) Dphi(label_N - ..) |
                        // | 0 ........0 Dphi(label_N - ...) |

                        // shifting the position
                        pos +=(this->num_classes_+1) + 1 ;


                    }
                    // boundary conditions
                    Dtype scalar_first_row = Dphi(this->theta_.cpu_data()[1] - Xw.cpu_data()[i]);
                    Df_t.mutable_cpu_data()[1] = scalar_first_row ;

                    Dtype scalar_last_row = Dphi(this->theta_.cpu_data()[this->num_classes_ - 1] - Xw.cpu_data()[i]);
                    Df_t.mutable_cpu_data()[(this->num_classes_+1)*(this->num_classes_ ) - 2 ] = -scalar_last_row ;


                    // transform from theta to gamma representation
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_classes_, this->num_classes_-1, this->num_classes_+1 , (Dtype)1.,
                                          Df_t.cpu_data(), Dtheta_2_Dgamma.cpu_data(), (Dtype)0.,
                                          Df_g.mutable_cpu_data());


                    // add: matrix * vector (tranposed!!)
                    // Operation: accumulate Dt_g * delta => SZ: (this->num_classes-1)
                    // Dt_g SZ: (this->num_classes_) x (this->num_classes_-1)
                    // top[0]->cpu_diff() SZ: (num) x (this->num_classes_)
                    // delta SZ: (this->num_classes_)

                    caffe_cpu_gemv<Dtype>(CblasTrans, this->num_classes_, this->num_classes_-1, (Dtype)1., Df_g.cpu_data(),
                                          &top[0]->cpu_diff()[i*this->num_classes_], (Dtype)1.,
                                          this->blobs_[1]->mutable_cpu_diff());

                }



            }



            // gradient w.r.t. bottom data
            // Differential matrix: Df / Dx : [num rows (batch size),  dim cols (dim of data) )]


            for (int i = 0; i < num; ++i) {

                //memset(Df_x, 0, this->num_classes_*dim*sizeof(Dtype));
                caffe_memset( this->num_classes_*dim*sizeof(Dtype), 0, Df_x.mutable_cpu_data() );


                for (int label=1; label<=this->num_classes_; ++label) {

                    // (phi(x) - phi(y))' => Dphi(x)-Dphi(y)
                    // equivalent to (and more numerically stable):
                    // (1 - phi(x) - phi(y))


                    Dtype scalar = (Dphi(this->theta_.cpu_data()[label] - Xw.cpu_data()[i]) - Dphi(this->theta_.cpu_data()[label-1] - Xw.cpu_data()[i]));

                    // Operation: scalar * W + Df_x -> Df_x
                    // W: this->blobs_[0]->cpu_data() SZ: dim
                    // Df_x SZ: (this->num_classes_) x (dim)
                    caffe_axpy(dim, -scalar, this->blobs_[0]->cpu_data(), &Df_x.mutable_cpu_data()[(label-1)*dim]);
                }
                // Gradient with respect to bottom data (transposed)
                // Operation: Df_x^T * delta : => OUT: (dim)
                // Df_x SZ : (this->num_classes_) x (dim)
                // delta : ( num ) x ( this->num_classes)
                caffe_cpu_gemv<Dtype>(CblasTrans, this->num_classes_, dim, (Dtype)1.,  Df_x.cpu_data(),
                                      &top[0]->cpu_diff()[i*this->num_classes_], (Dtype)0.,
                                      &bottom[0]->mutable_cpu_diff()[i*dim]);
            }


            //delete[] Df_x;

            //delete[] Xw;

        }
    }




    INSTANTIATE_CLASS(OrdinalRegressionLayer);
    REGISTER_LAYER_CLASS(OrdinalRegression);

}  // namespace caffe
