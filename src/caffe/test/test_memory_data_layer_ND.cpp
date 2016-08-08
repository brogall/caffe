#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "caffe/layers/memory_dataND_layer.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

    template <typename TypeParam>
    class MemoryDataNDLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;

    protected:
        MemoryDataNDLayerTest()
        : data_(new Blob<Dtype>()),
        labels_(new Blob<Dtype>()),
        data_blob_(new Blob<Dtype>()),
        label_blob_(new Blob<Dtype>()) {}
        virtual void SetUp() {
            batch_size_ = 8;
            batches_ = 12;
            channels_ = 4;
            height_ = 7;
            width_ = 11;
            depth_ = 5;

            blob_top_vec_.push_back(data_blob_);
            blob_top_vec_.push_back(label_blob_);
            // pick random input data
            FillerParameter filler_param;
            GaussianFiller<Dtype> filler(filler_param);
            std::vector<int> shape_vector_(5);
            shape_vector_[0] = batches_ * batch_size_;
            shape_vector_[1] = channels_;
            shape_vector_[2] = height_;
            shape_vector_[3] = width_;
            shape_vector_[4] = depth_;


            data_->Reshape(shape_vector_);
            for (int i=1; i<5; i++) {
                shape_vector_[i] = 1;
            }
            labels_->Reshape(shape_vector_);
            filler.Fill(this->data_);
            filler.Fill(this->labels_);
        }

        virtual ~MemoryDataNDLayerTest() {
            delete data_blob_;
            delete label_blob_;
            delete data_;
            delete labels_;
        }
        int batch_size_;
        int batches_;
        int channels_;
        int height_;
        int width_;
        int depth_;
        // we don't really need blobs for the input data, but it makes it
        //  easier to call Filler
        Blob<Dtype>* const data_;
        Blob<Dtype>* const labels_;
        // blobs for the top of MemoryDataLayer
        Blob<Dtype>* const data_blob_;
        Blob<Dtype>* const label_blob_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
    };

    TYPED_TEST_CASE(MemoryDataNDLayerTest, TestDtypesAndDevices);

    TYPED_TEST(MemoryDataNDLayerTest, TestSetup) {
        typedef typename TypeParam::Dtype Dtype;

        LayerParameter layer_param;
        layer_param.add_top("data");
        layer_param.add_top("label");
        MemoryDataNDParameter* md_param = layer_param.mutable_memory_datand_param();
        md_param->set_batch_size(this->batch_size_);
        //md_param->add_size(this->batch_size_);
        md_param->add_size(this->channels_);
        md_param->add_size(this->height_);
        md_param->add_size(this->width_);
        md_param->add_size(this->depth_);
        md_param->add_classes(1);
        shared_ptr<Layer<Dtype> > layer(
                                        new MemoryDataNDLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        EXPECT_EQ(this->data_blob_->shape(0), this->batch_size_);
        EXPECT_EQ(this->data_blob_->shape(1), this->channels_);
        EXPECT_EQ(this->data_blob_->shape(2), this->height_);
        EXPECT_EQ(this->data_blob_->shape(3), this->width_);
        EXPECT_EQ(this->data_blob_->shape(4), this->depth_);
        EXPECT_EQ(this->label_blob_->shape(0), this->batch_size_);
    }



    // run through a few batches and check that the right data appears
    TYPED_TEST(MemoryDataNDLayerTest, TestForward) {
        typedef typename TypeParam::Dtype Dtype;

        LayerParameter layer_param;
        MemoryDataNDParameter* md_param = layer_param.mutable_memory_datand_param();
        md_param->set_batch_size(this->batch_size_);
        //md_param->add_size(this->batch_size_);
        md_param->add_size(this->channels_);
        md_param->add_size(this->height_);
        md_param->add_size(this->width_);
        md_param->add_size(this->depth_);
        md_param->add_classes(1);
        shared_ptr<MemoryDataNDLayer<Dtype> > layer(
                                                    new MemoryDataNDLayer<Dtype>(layer_param));
        layer->DataLayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);

        vector<Dtype* > tmp_vec;
        tmp_vec.push_back(this->labels_->mutable_cpu_data());
        
        vector<int> tmp_classes;
        tmp_classes.push_back(1);

        layer->Reset(this->data_->mutable_cpu_data(),
                     &tmp_vec, this->data_->shape(0), &tmp_classes);
        std::vector<int> offsetVec_(5);
        offsetVec_[0] = 1;

        for (int i = 0; i < this->batches_ * 6; ++i) {
            int batch_num = i % this->batches_;
            layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
            for (int j = 0; j < this->data_blob_->count(); ++j) {
                EXPECT_EQ(this->data_blob_->cpu_data()[j],
                          this->data_->cpu_data()[
                                                  this->data_->offset(offsetVec_) * this->batch_size_ * batch_num + j]);
            }
            for (int j = 0; j < this->label_blob_->count(); ++j) {
                EXPECT_EQ(this->label_blob_->cpu_data()[j],
                          this->labels_->cpu_data()[this->batch_size_ * batch_num + j]);
            }
        }
    }

   
}  // namespace caffe
