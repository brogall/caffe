#ifndef CAFFE_MEMORY_DATAND_LAYER_HPP_
#define CAFFE_MEMORY_DATAND_LAYER_HPP_

#include <vector>
#include "boost/shared_array.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
    template <typename Dtype>
    class MemoryDataNDLayer : public BaseDataLayer<Dtype> {
    public:
        explicit MemoryDataNDLayer(const LayerParameter& param)
        : BaseDataLayer<Dtype>(param), has_new_data_(false) { top_count_ = param.top_size(); data_ = 0; }
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const { return "MemoryDataND"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return top_count_; }
        virtual ~MemoryDataNDLayer();
        
        //virtual void AddDatumVector(const vector<Datum>& datum_vector);
        //virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
        //                         const vector<int>& labels);
        
        // Reset should accept const pointers, but can't, because the memory
        //  will be given to Blob, which is mutable
        void Reset(Dtype* data, vector<Dtype* > *, int n, vector<int>* num_classes);
        void set_batch_size(int new_size);
        
        int batch_size() { return batch_size_; }
        inline int num_axes() const { return shape_.size(); }
        inline const vector<int>& shape() const { return shape_; }
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        
        int  size_;
        Dtype* data_;
        vector<boost::shared_array<Dtype> > labels_;
        int n_;
        int batch_size_;
        vector<int> num_classes_;
        int top_count_;
        size_t pos_;
        vector<int> shape_;
        Blob<Dtype> added_data_;
        Blob<Dtype> added_label_;
        bool has_new_data_;
    };


}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
