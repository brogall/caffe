#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ordinal_regression_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

    //#ifndef CPU_ONLY
    //    extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
    //#endif

    template <typename TypeParam>
    class OrdinalRegressionLayerTest : public ::testing::Test {
        // typedef typename TypeParam::TypeParam TypeParam;
    protected:
        OrdinalRegressionLayerTest()
        : blob_bottom_(new Blob<TypeParam>(7, 3, 1, 1)),
        blob_top_(new Blob<TypeParam>()) {
            // fill the values
            FillerParameter filler_param;
            UniformFiller<TypeParam> filler(filler_param);
            /*filler_param.set_std(1);
             GaussianFiller<TypeParam> filler(filler_param);
             for (int i=0; i<60*7; i++) {
             this->blob_bottom_->mutable_cpu_data()[i]=float(i);
             }*/
            filler.Fill(this->blob_bottom_);
            for (int i = 0; i < 7 * 3; i ++)
            {
                this->blob_bottom_->mutable_cpu_data()[i] = TypeParam(i);
            }
            blob_bottom_vec_.push_back(blob_bottom_);
            blob_top_vec_.push_back(blob_top_);
        }
        virtual ~OrdinalRegressionLayerTest() { delete blob_bottom_; delete blob_top_; }
        Blob<TypeParam>* const blob_bottom_;
        Blob<TypeParam>* const blob_top_;
        vector<Blob<TypeParam>*> blob_bottom_vec_;
        vector<Blob<TypeParam>*> blob_top_vec_;
    };

    TYPED_TEST_CASE(OrdinalRegressionLayerTest, TestDtypes);


    TYPED_TEST(OrdinalRegressionLayerTest, TestSetUp) {

        LayerParameter layer_param;
        int num_classes = 5;
        OrdinalRegressionParameter* ordinal_regression_param = layer_param.mutable_ordinal_regression_param();
        ordinal_regression_param->set_classes(num_classes);

        shared_ptr<OrdinalRegressionLayer<TypeParam> > layer(
                                                                     new OrdinalRegressionLayer<TypeParam>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        EXPECT_EQ(this->blob_top_->shape(0), 7);
        EXPECT_EQ(this->blob_top_->shape(1), 5);
    }


    TYPED_TEST(OrdinalRegressionLayerTest, TestForward) {

        int num_classes = 5;
        LayerParameter layer_param;
        OrdinalRegressionParameter* ordinal_regression_param = layer_param.mutable_ordinal_regression_param();
        ordinal_regression_param->set_classes(num_classes);
        shared_ptr<OrdinalRegressionLayer<TypeParam> > layer(new OrdinalRegressionLayer<TypeParam>(layer_param));
        layer->blobs().resize(2);
        //layer->blobs()[0].reset(new Blob<TypeParam>(5,1,0,0));

        vector<int> blob1_shape(1);
        blob1_shape[0] = 3;
        layer->blobs()[0].reset(new Blob<TypeParam>(blob1_shape));

        TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();

        weights[0] = 0.5;
        weights[1] = -0.3;
        weights[2] =  0.2;
        //weights[3] = 0.7;
        //weights[4] =  -0.4;


        vector<int> blob2_shape(1);
        blob2_shape[0] = num_classes-1;

        layer->blobs()[1].reset(new Blob<TypeParam>(blob2_shape));
        TypeParam* gamma = layer->blobs()[1]->mutable_cpu_data();

        gamma[0] = -.1;
        gamma[1] = .2;
        gamma[2] = .7;
        gamma[3] = .3;


        Caffe::set_mode(Caffe::CPU);
        //LogisticOrdinalRegressionLossLayer<TypeParam> layer(layer_param);
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);;
        EXPECT_EQ(this->blob_top_->channels(),num_classes);
        EXPECT_EQ(this->blob_top_->num(), 7);
        layer->updateTheta_cpu();


        for (int i = 0; i < 7 * 3; i ++)
        {
            this->blob_bottom_->mutable_cpu_data()[i] = TypeParam(i);
        }
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        //TypeParam epsilon = 1.0E-1;
        //EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], 128.9, epsilon);
    }



    TYPED_TEST(OrdinalRegressionLayerTest, UpdateThetaCPU) {

        int num_classes = 5;
        LayerParameter layer_param;
        OrdinalRegressionParameter* ordinal_regression_param = layer_param.mutable_ordinal_regression_param();
        ordinal_regression_param->set_classes(num_classes);
        shared_ptr<OrdinalRegressionLayer<TypeParam> > layer(new OrdinalRegressionLayer<TypeParam>(layer_param));
        layer->blobs().resize(3);
        //layer->blobs()[0].reset(new Blob<TypeParam>(5,1,0,0));

        vector<int> blob1_shape(1);
        blob1_shape[0] = 3;
        layer->blobs()[0].reset(new Blob<TypeParam>(blob1_shape));

        TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();

        weights[0] = 0.5;
        weights[1] = -0.3;
        weights[2] =  0.2;
        //weights[3] = 0.7;
        //weights[4] =  -0.4;


        vector<int> blob2_shape(1);
        blob2_shape[0] = num_classes-1;

        layer->blobs()[1].reset(new Blob<TypeParam>(blob2_shape));
        TypeParam* gamma = layer->blobs()[1]->mutable_cpu_data();

        gamma[0] = 1;
        gamma[1] = 2.;
        gamma[2] = 3.;
        gamma[3] = 4.;


        Caffe::set_mode(Caffe::CPU);
        //LogisticOrdinalRegressionLossLayer<TypeParam> layer(layer_param);
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->updateTheta_cpu();


        const TypeParam* res = layer->getTheta();

        //CHECK_EQ(res[0], 1.0); // min value
        CHECK_EQ(res[1], 1.0); //13
        CHECK_EQ(res[2], 5.0); // 0
        CHECK_EQ(res[3], 14.0); //1
        CHECK_EQ(res[4], 30.0); //3
    }



    TYPED_TEST(OrdinalRegressionLayerTest, TestGradient) {


        int num_classes = 5;
        LayerParameter layer_param;
        OrdinalRegressionParameter* ordinal_regression_param = layer_param.mutable_ordinal_regression_param();
        ordinal_regression_param->set_classes(num_classes);
        OrdinalRegressionLayer<TypeParam> layer(layer_param);
        layer.blobs().resize(2);


        vector<int> blob1_shape(1);
        blob1_shape[0] = 24;
        layer.blobs()[0].reset(new Blob<TypeParam>(blob1_shape));

        TypeParam* weights = layer.blobs()[0]->mutable_cpu_data();

        for (int i=0; i<3; i++) {

            weights[i] = float(i+1.0); }



        vector<int> blob2_shape(1);
        blob2_shape[0] = num_classes-1;

        layer.blobs()[1].reset(new Blob<TypeParam>(blob2_shape));
        TypeParam* gamma = layer.blobs()[1]->mutable_cpu_data();

        gamma[0] = -.1;//-0.1;
        gamma[1] = .2;//0.1;
        gamma[2] = .7;//0.2;
        gamma[3] = .3;//0.3;







        //layer.blobs().resize(2);
        GradientChecker<TypeParam> checker(1e-1, 2e-2, 1701);//, 1, 0.01);//, 0, 0.01);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                        this->blob_top_vec_);

    }

}  // namespace caffe
