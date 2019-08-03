# SLAESR
Separating latent automatic encoder super-resolution network.  

Train:  
dataset: https://github.com/ANIME305/Anime-GAN-tensorflow#open-sourced-dataset  
loss: ssim & l1  
train_resolution: 64x64 -> 128x128  

Although it is trained at 64->128 resolution, it is also effective for large pictures in the test.  

Because it is only trained on the anime avatar data set, it may not work well for real images.  

There are two models, the difference is only the upsampling layer.  
Model 1 is upsampled using 2x2 deconvolution.  
Model 2 is upsampled using 3x3 convolution and bilinear.  

The two models generate the difference in the image.  
The image generated by Model 1 has a higher saturation, but there are many artifacts that I can't remove in many ways.  
(including replacement deconvolution to 3x3 convolution + PixelShuffle)  
The image generated by Model 2 has a lower saturation than the original image, but the texture is very smooth.  
You can take a closer look at the sr_dst_1 and sr_dst_2 folders.  

# Dependent
pytorch 1.1  
numpy  
imageio  
opencv-python  

# Training Example

### model_1:  
64->128  
![](https://github.com/One-sixth/SLAESR/blob/master/samples_1/test_32100.jpg)
128->256, only test output. No train.
![](https://github.com/One-sixth/SLAESR/blob/master/samples_1/test_32100_SR.jpg)

### model_2:  
64->128  
![](https://github.com/One-sixth/SLAESR/blob/master/samples_2/test_41400.jpg)
128->256, only test output. No train.
![](https://github.com/One-sixth/SLAESR/blob/master/samples_2/test_41400_SR.jpg)

# Testing Example

### origin:  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_src/1.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_src/2.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_src/3.png)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_src/%E3%81%95%E3%82%88%E3%81%AA%E3%82%89%E3%81%AE%E6%9C%9D%E3%81%AB%E7%B4%84%E6%9D%9F%E3%81%AE%E8%8A%B1%E3%82%92%E3%81%8B%E3%81%96%E3%82%8D%E3%81%86.jpg)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_src/%E5%B0%86%E6%89%80%E6%9C%89%E7%9A%84%E6%AD%8C%E7%8C%AE%E7%BB%99%E6%9C%AA%E6%9D%A5%E7%9A%84%E4%BD%A0.png)

### model_1:  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_1/1.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_1/2.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_1/3.png)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_1/%E3%81%95%E3%82%88%E3%81%AA%E3%82%89%E3%81%AE%E6%9C%9D%E3%81%AB%E7%B4%84%E6%9D%9F%E3%81%AE%E8%8A%B1%E3%82%92%E3%81%8B%E3%81%96%E3%82%8D%E3%81%86.jpg)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_1/%E5%B0%86%E6%89%80%E6%9C%89%E7%9A%84%E6%AD%8C%E7%8C%AE%E7%BB%99%E6%9C%AA%E6%9D%A5%E7%9A%84%E4%BD%A0.png)  

### model_2:  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_2/1.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_2/2.png)
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_2/3.png)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_2/%E3%81%95%E3%82%88%E3%81%AA%E3%82%89%E3%81%AE%E6%9C%9D%E3%81%AB%E7%B4%84%E6%9D%9F%E3%81%AE%E8%8A%B1%E3%82%92%E3%81%8B%E3%81%96%E3%82%8D%E3%81%86.jpg)  
![](https://github.com/One-sixth/SLAESR/blob/master/sr_dst_2/%E5%B0%86%E6%89%80%E6%9C%89%E7%9A%84%E6%AD%8C%E7%8C%AE%E7%BB%99%E6%9C%AA%E6%9D%A5%E7%9A%84%E4%BD%A0.png)  


# Enhance your image
You can find the pre-training model here.
https://github.com/One-sixth/SLAESR/releases
Download model_1.7z and model_2.7z to the source code directory and extract them.

if you want to try model_1  
push your image in sr_src dir.  
just run  
```
python3 _test_ae_sr_1.py
```
You can see the high resolution image output in sr_dst_1 dir  

if you want to try model_2  
just run  
```
python3 _test_ae_sr_2.py
```
The output image is in sr_dst_2.  


# Train on you dataset
For Model 1  
Delete all weight.  
Edit file _train_ae_sr_1.py  
change  
```
dataset_path = r'../datasets/getchu_aligned_with_label/GetChu_aligned2'
```
to  
```
dataset_path = r'your/datasets/path/'
```
save file  
run  
```
python3 _train_ae_sr_1.py  
```
Then the training will be start.  

Model 2 operations are similar to the above.  

# Network Architecture
wait to add...  
