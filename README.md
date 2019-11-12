# Splash of Colour  
Eng: Colour Splash Effect  
Kor: 컬러추출 필터  
Chn: 局部彩色  

This is a technique that remains specific colour tone while making every others monochrome.  
Main theme of my implementation this time, is the use of Bilateral K-means clustering (which simply means k = 2) after RGB -> YCbCr Conversion.

它能让你将图片转化为黑白色，然后再在指定的区域增添鲜艳的色彩，这样能够制作出独具特色的照片出来。
  
Example 1.  
![SoC_Ex_1](https://github.com/koominsoo/Splash_of_Colour/blob/master/result/SoC_example.png)
