# Splash of Colour through Binary Segmentation

In Korean, it's 컬러추출필터 and in Chinese 局部彩色. 

This is a technique that remains specific colour tone while making every others monochrome.  
Main theme of my implementation is use of Bilateral K-means clustering (which simply means k = 2) after RGB -> YCbCr Conversion. For K-means clustering, I didn't consider Y (luminence value).
  
Demo Result  
![SoC_Ex_1](https://github.com/koominsoo/Splash_of_Colour/blob/master/result/SoC_example.png)
