#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/layers/memory_dataND_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

    template <typename Dtype>
    void MemoryDataNDLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
        int num_of_axes_ = this->layer_param_.memory_datand_param().size_size();

               batch_size_ = this->layer_param_.memory_datand_param().batch_size();

        size_ = 1;
        //std::cout << "NOA: " << num_of_axes_ << std::endl;
        shape_.resize(num_of_axes_+1);
        for(int i=0; i<num_of_axes_;i++)
        {
            shape_[i+1] = this->layer_param_.memory_datand_param().size(i);
            //if ( i > 0 )
            size_ *= shape_[i+1];
            //std::cout << "shape "<<i<<": " << shape_[i] << std::endl;
        }
        CHECK_GT(batch_size_ * size_, 0) <<
        "batch_size, channels, height, and width must be specified and"
        " positive in memory_data_param";

        top_count_ = this->layer_param_.top_size();

        vector<int> tmp_vec_(num_of_axes_+1);
        for(int i=1; i<num_of_axes_+1;i++)
        {
            tmp_vec_[i] = shape_[i];
        }
        tmp_vec_[0] = batch_size_;
        vector<int> label_shape(2);

        top[0]->Reshape(tmp_vec_);
        for(int i=1;i<top.size();++i) {
            label_shape[0] = batch_size_;
            label_shape[1] = this->layer_param_.memory_datand_param().classes(i-1);
        top[i]->Reshape(label_shape);}
        this->top_count_ = top.size();
        //added_data_.Reshape(shape_);
        //added_label_.Reshape(label_shape);
        //data_ = NULL;
        //labels_ = NULL;
        data_ = NULL;
        //labels_ = NULL;
        //added_data_.cpu_data();
        //added_label_.cpu_data();
    }
    /*
     template <typename Dtype>
     void MemoryDataNDLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
     CHECK(!has_new_data_) <<
     "Can't add data until current data has been consumed.";
     size_t num = datum_vector.size();
     CHECK_GT(num, 0) << "There is no datum to add.";
     CHECK_EQ(num % shape_[0], 0) <<
     "The added data must be a multiple of the batch size.";
     shape_[0] = num;
     added_data_.Reshape(shape_);
     std::vector<int> tmp_vec_ (1);
     tmp_vec_[0] = num;

     added_label_.Reshape(tmp_vec_);
     // Apply data transformations (mirror, scale, crop...)
     this->data_transformer_->Transform(datum_vector, &added_data_);
     // Copy Labels
     Dtype* top_label = added_label_.mutable_cpu_data();
     for (int item_id = 0; item_id < num; ++item_id) {
     top_label[item_id] = datum_vector[item_id].label();
     }
     // num_images == batch_size_
     Dtype* top_data = added_data_.mutable_cpu_data();
     Reset(top_data, top_label, num);
     has_new_data_ = true;
     }

     template <typename Dtype>
     void MemoryDataNDLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
     const vector<int>& labels) {
     size_t num = mat_vector.size();
     CHECK(!has_new_data_) <<
     "Can't add mat until current data has been consumed.";
     CHECK_GT(num, 0) << "There is no mat to add";
     CHECK_EQ(num % shape_[0], 0) <<
     "The added data must be a multiple of the batch size.";

     std::vector<int> tmp_vec_ (1);
     tmp_vec_[0] = num;
     added_label_.Reshape(tmp_vec_);
     // Apply data transformations (mirror, scale, crop...)
     this->data_transformer_->Transform(mat_vector, &added_data_);
     // Copy Labels
     Dtype* top_label = added_label_.mutable_cpu_data();
     for (int item_id = 0; item_id < num; ++item_id) {
     top_label[item_id] = labels[item_id];
     }
     // num_images == batch_size_
     Dtype* top_data = added_data_.mutable_cpu_data();
     Reset(top_data, top_label, num);
     has_new_data_ = true;
     }
     */
    template <typename Dtype>
    void MemoryDataNDLayer<Dtype>::Reset(Dtype* data, vector<Dtype* > *labels, int n, vector<int>* num_classes) {
        CHECK(data);
        CHECK(labels);
        CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
        // Warn with transformation parameters since a memory array is meant to
        // be generic and no transformations are done with Reset().
        if (this->layer_param_.has_transform_param()) {
            LOG(WARNING) << this->type() << " does not transform array data on Reset()";
        }
        //data_ = data;
        //labels_ = labels;

        if ( data_)
            delete[] data_;

        if (labels_.size() > 0 )
            labels_.clear();
        
        if (num_classes_.size() > 0 )
            num_classes_.clear();

        data_ = new Dtype[n * size_];



        memcpy(data_, data, sizeof(Dtype) * n * size_);
        for(int i=0; i<labels->size();++i) {
            boost::shared_array<Dtype> tmp_(new Dtype[n * num_classes->at(i)]);
            memcpy(tmp_.get(), labels->at(i), sizeof(Dtype) * n * num_classes->at(i));
            labels_.push_back(tmp_);
            num_classes_.push_back(num_classes->at(i));
            }

        n_ = n;
        pos_ = 0;
        for(int i=0; i<labels->size();++i) {
            CHECK_EQ(this->layer_param_.memory_datand_param().classes(i), num_classes->at(i)) << "Number of classes if input and defnition file have to match. " << i; }
    }

    template <typename Dtype>
    void MemoryDataNDLayer<Dtype>::set_batch_size(int new_size) {
        /*CHECK(!has_new_data_) <<
        "Can't change batch_size until current data has been consumed.";
        batch_size_ = new_size;
        std::vector<int> tmp_vec1_ = shape_;
        tmp_vec1_[0] = batch_size_;
        added_data_.Reshape(shape_);
        std::vector<int> tmp_vec2_ (2);
        tmp_vec2_[0] = batch_size_;
        tmp_vec2_[1] = num_classes_;
        added_label_.Reshape(tmp_vec2_);*/
    }

    template <typename Dtype>
    MemoryDataNDLayer<Dtype>:: ~MemoryDataNDLayer()
    {
        if (data_)
            delete[] data_;

        if (labels_.size() > 0 )
            labels_.clear();
        if (num_classes_.size() > 0 )
            num_classes_.clear();
    }

    template <typename Dtype>
    void MemoryDataNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
        CHECK(data_) << "MemoryDataNDLayer needs to be initalized by calling Reset";
        //for (int i=0; i<4; i++) {
        //    std::cout << "S"<<i<<": " << bottom[0]->shape(i) << std::endl; }
        std::vector<int> tmp_vec1_ = shape_;
        tmp_vec1_[0]  = batch_size_;
        top[0]->Reshape(tmp_vec1_);

        top[0]->set_cpu_data(data_ + pos_ * size_);
        for(int i=1;i<top.size();++i) {
            std::vector<int> tmp_vec2_ (2);
            tmp_vec2_[0] = batch_size_;
            tmp_vec2_[1] = num_classes_[i-1];
            
            top[i]->Reshape(tmp_vec2_);
            top[i]->set_cpu_data(labels_[i-1].get() + pos_);
            }
        pos_ = (pos_ + batch_size_) % n_;
        if (pos_ == 0)
        has_new_data_ = false;
    }

    INSTANTIATE_CLASS(MemoryDataNDLayer);
    REGISTER_LAYER_CLASS(MemoryDataND);

}  // namespace caffe
