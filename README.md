# Post-training Quantization

Paper: ["Post-training Quantization for Neural Networks with Provable Guarantees"](https://arxiv.org/abs/2201.11113).

## Obtaining ImageNet Dataset

In this project, we make use of the Imagenet dataset, 
in particular, we use the ILSVRC-2012 version. 

To obtain the Imagenet dataset, one can submit a request through this [link](https://image-net.org/request).

Once the dataset is obtained, place the `.tar` files for training set and validation set both under the `data/ILSVRC2012` directory of this repo. 

Then use the following procedure to unzip Imagenet dataset:
```
# Extract the training data and move images to subfolders:
mkdir ILSVRC2012_img_train
mv ILSVRC2012_img_train.tar ILSVRC2012_img_train 
cd ILSVRC2012_img_train 
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

# Extract the validation data:
cd ..
mkdir ILSVRC2012_img_val
mv ILSVRC2012_img_val.tar ILSVRC2012_img_val && cd ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
``` 

## Run the code

For example, if we want to quantize the ResNet-18 using ImageNet data with bit = 4, batch_size = 256, scalar = 1.16, then we can try this:

`python src/main.py -b 4 -bs 256 -s 1.16`

There are other options we can select, see `main.py`.
