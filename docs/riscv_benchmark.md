## RISCV benchmark

### 1. Compilation

Please refer to [RISCV Platform Guide](docs/riscv_usage.md).

### 2. Running benchmark

Please refer to [RISCV Platform Guide](docs/riscv_usage.md).

### 3. Testing configuration

* Information of mathines:
  - Board: AllWinner D1
  - CPU: XuanTie C906 (1 core, 1GHz)
  - Memory: 2 GB
  - OS: Debian by PerfXLab
* Toolchains:
  - [Toolchain1: toolchain package for OpenCV v4.4](https://occ.t-head.cn/community/download?id=3996672928124047360)
  - [Toolchain2: toolchain package for OpenCV v4.5.5](https://github.com/damonyu1989/Tools)
* Data from PPLCV
  Since the toolchain package for OpenCV v4.5.5 cannot compile the intrinsics used by PPLCV, the test data is tested by the toolchain package for OpenCV v4.4.
* Data from OpenCV
  In order to control variables and ensure the unified toolchain, we tested the performance data from OpenCV based on the toolchain package for OpenCV v4.4; in addition, since the toolchain package for OpenCV v4.4 is not optimized for the RVV vector instruction set, we also tested the performance data from OpenCV based on the toolchain package for OpenCV v4.5.5 with RVV optimization.

### 4. Speedup statistics

Due to the diversity of input image sizes, we test multiple cases with different image sizes.

In each case with fixed image size, there are three kinds of data, which are the data tested by PPLCV with Toolchain1, the data tested by OpenCV with Toolchain1 and the data tested by OpenCV with Toolchain2. They all run on a serial of parameter combinations covering common usage and the elapsed time is recorded. Besides the particular parameters of a funciton, the supported data types(uchar/float) and the channels(1/3/4) are tested for each function.

We describe performance in terms of fames per second(fps).

The illustration about tabular data:
* (xx,xx,xx)(uchar) : (channel-1,channel-3,channel-4)(uchar)
* (xx,xx,xx)(float) : (channel-1,channel-3,channel-4)(float)
* (xx)(uchar/float) : Some operators have the fixed channel implementation, so there is only one result in brackets.
* (--,--,--)(uchar/float) : Due to the illegal instructions error during compilation with Toolchain2 in OpenCV, the performance data of some operators cannot be obtained.

#### 4.1. Image Size: 320*240 

| function | PPLCV | OpenCV v4.4 | OpenCV v4.5.5 |
| :-------: | :-------: | :-------: | :-------: |
| Add | (7025.57,1503.77,645.433)(uchar)<br>(2242.89,729.383,543.484)(float) | (1697.23,568.16,425.678)(uchar)<br>(799.562,264.425,198.817)(float) | (2375.86,797.43,553.52)(uchar)<br>(519.93,184.80,134.00)(float) |
| AddWeighted | (3106.88,1000.61,754.552)(uchar)<br>(1297.78,447.633,300.76)(float) | (506.68,169.254,126.818)(uchar)<br>(686.836,227.614,165.973)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Subtract | (7023.42,1490.48,647.812)(uchar)<br>(2237.95,721.617,546.056)(float) | (1693.54,557.489,424.135)(uchar)<br>(800.871,265.577,199.112)(float) | (2368.00,795.082,560.073)(uchar)<br>(554.764,178.787,134.273)(float) |
| Mul | (6651.76,1417.04,614.72)(uchar)<br>(2251.88,722.524,546.221)(float) | (1208.11,405.246,304.114)(uchar)<br>(798.71,265.548,198.62)(float) | (2121.32,710.781,510.922)(uchar)<br>(554.221,179.631,132.901)(float) |
| Div | (3280.87,1097.92,712.847)(uchar)<br>(920.42,308.232,228.638)(float) | (307.395,102.53,76.9158)(uchar)<br>(454.674,150.205,110.769)(float) | (599.243,200.188,150.005)(uchar)<br>(420.187,140.094,104.80)(float) |
| BGR2BGRA | (1807.62)(uchar)<br>(811.11)(float) | (1083.7)(uchar)<br>(830.37)(float) | (3731.82)(uchar)<br>(991.22)(float) |
| BGRA2BGR | (1787.94)(uchar)<br>(605.86)(float) | (1166.65)(uchar)<br>(909.73)(float) | (3369.75)(uchar)<br>(948.99)(float) |
| BGR2GRAY | (2045.9)(uchar)<br>(1893.48)(float) | (741.83)(uchar)<br>(465.83)(float) | (645.61)(uchar)<br>(1199.06)(float) |
| BGRA2GRAY | (1977.18)(uchar)<br>(1639.55)(float) | (738.96)(uchar)<br>(458.09)(float) | (631.94)(uchar)<br>(1041.75)(float) |
| GRAY2BGR | (2328.55)(uchar)<br>(1772.8)(float) | (935.61)(uchar)<br>(338.65)(float) | (6526.73)(uchar)<br>(1882.83)(float) |
| GRAY2BGRA | (792.56)(uchar)<br>(267.76)(float) | (739.10)(uchar)<br>(261.21)(float) | (6190.61)(uchar)<br>(1567.43)(float) |
| NV2BGR<br>(NV122RGB_MODE) | (448.274)(uchar)<br>(-)(float) | (231.402)(uchar)<br>(-)(float) | (417.745)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV122BGR_MODE) | (448.294)(uchar)<br>(-)(float) | (230.561)(uchar)<br>(-)(float) | (436.692)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212RGB_MODE) | (450.548)(uchar)<br>(-)(float) | (231.532)(uchar)<br>(-)(float) | (417.771)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212BGR_MODE) | (450.599)(uchar)<br>(-)(float) | (230.877)(uchar)<br>(-)(float) | (437.199)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2I420_MODE) | (677.007)(uchar)<br>(-)(float) | (451.565)(uchar)<br>(-)(float) | (792.686)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2I420_MODE) | (676.875)(uchar)<br>(-)(float) | (483.716)(uchar)<br>(-)(float) | (807.86)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2YV12_MODE) | (677.485)(uchar)<br>(-)(float) | (451.452)(uchar)<br>(-)(float) | (787.36)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2YV12_MODE) | (677.011)(uchar)<br>(-)(float) | (484.042)(uchar)<br>(-)(float) | (802.806)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202RGB_MODE) | (434.6)(uchar)<br>(-)(float) | (228.633)(uchar)<br>(-)(float) | (416.665)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202BGR_MODE) | (435.145)(uchar)<br>(-)(float) | (229.485)(uchar)<br>(-)(float) | (418.502)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122RGB_MODE) | (433.781)(uchar)<br>(-)(float) | (229.887)(uchar)<br>(-)(float) | (415.477)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122BGR_MODE) | (434.371)(uchar)<br>(-)(float) | (229.446)(uchar)<br>(-)(float) | (418.732)(uchar)<br>(-)(float) |
| CopyMakeborder<br>(BORDER_CONSTANT) | (3707.6600,2335.6800,3557.6200)(uchar)<br>(2232.1600,1010.9000,828.5250)(float) | (5799.5900,3068.3100,2507.9700)(uchar)<br>(2511.4800,1046.7500,888.7030)(float) | (5832.1400,2988.2800,2579.3900)(uchar)<br>(2588.3500,1046.6400,869.0830)(float) |
| CopyMakeborder<br>(BORDER_REPLICATE) | (3445.5800,2247.7900,1942.0500)(uchar)<br>(2205.5500,1001.3300,793.1180)(float) | (3586.3500,2221.6700,2209.4400)(uchar)<br>(2210.8700,987.5970,795.6220)(float) | (3635.3600,2173.2800,2211.3100)(uchar)<br>(2201.1200,978.1700,780.1600)(float) |
| CopyMakeborder<br>(BORDER_REFLECT) | (3439.9200,2237.3500,1940.9300)(uchar)<br>(2195.2400,998.5460,788.7660)(float) | (3595.1000,2221.0600,2209.1000)(uchar)<br>(2207.0100,984.2030,793.4850)(float) | (3629.1500,2163.9100,2200.1200)(uchar)<br>(2185.4300,973.9730,776.6040)(float) |
| CopyMakeborder<br>(BORDER_REFLECT101) | (3447.1700,2246.1700,1942.2200)(uchar)<br>(2199.1100,998.1840,789.7650)(float) | (3601.6800,2214.1000,2208.9400)(uchar)<br>(2207.8300,982.5590,791.8290)(float) | (3633.1600,2165.7800,2208.6400)(uchar)<br>(2192.1500,974.0990,777.1230)(float) |
| Dilate<br>(k3x3) | (1653.3400,614.8140,453.8970)(uchar)<br>(547.2540,145.5130,109.9030)(float) | (412.2870,139.2300,105.2140)(uchar)<br>(218.2330,63.1068,45.8709)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Dilate<br>(k5x5) | (766.2550,317.7950,248.1010)(uchar)<br>(302.9960,86.7132,55.3967)(float) | (297.5410,99.6072,73.8433)(uchar)<br>(154.1220,43.8696,30.8186)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k3x3) | (1317.1700,620.4460,498.5980)(uchar)<br>(491.2800,200.3740,114.5190)(float) | (412.1360,138.4390,104.5760)(uchar)<br>(216.6890,63.0695,46.7089)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k5x5) | (691.79,323.80,263.31)(uchar)<br>(261.89,106.60,56.89)(float) | (297.61,99.29,73.59)(uchar)<br>(153.50,43.08,30.13)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Flip<br>(mode:-1) | (5459.33,1431.88,1162.93)(uchar)<br>(940.28,625.63,551.91)(float) | (1557.48,521.92,395.93)(uchar)<br>(395.83,123.02,90.00)(float) | (2591.59,611.82,637.37)(uchar)<br>(644.99,241.64,142.12)(float) |
| Flip<br>(mode:0) | (7759.32,5734.22,5115.27)(uchar)<br>(3332.59,1898.71,1626.56)(float) | (3222.87,1199.65,944.83)(uchar)<br>(941.24,334.71,232.36)(float) | (3175.35,1102.03,915.60)(uchar)<br>(913.18,330.11,224.47)(float) |
| Flip<br>(mode:1) | (4262.03,1423.02,1176.99)(uchar)<br>(935.67,625.55,550.31)(float) | (1999.53,720.07,506.04)(uchar)<br>(506.25,137.22,113.00)(float) | (3659.43,728.16,846.43)(uchar)<br>(848.66,288.50,182.89)(float) |
| Resize<br>(INTERPOLATION_NEAREST_LINEAR)<br>(320/240/640/480) | (247.29,81.15,75.67)(uchar)<br>(231.19,81,43,78.19)(float) | (118.02,39.77,29.37)(uchar)<br>(161.54,51.32,35.01)(float) | (206.68,47.43,36.23)(uchar)<br>(187.28,43.50,31,16)(float) |
| Resize<br>(INTERPOLATION_NEAREST_POINT)<br>(320/240/640/480) | (962.85,223.01,814.56)(uchar)<br>(749.58,162.78,127.38)(float) | (557.23,228.55,298.05)(uchar)<br>(296.52,158.83,76.49)(float) | (559.25,227.78,297.32)(uchar)<br>(297.07,160.50,27.90)(float) |
| Resize<br>(INTERPOLATION_AREA)<br>(320/240/640/480) | (247.88,81.38,75.51)(uchar)<br>(231.54,77.55,76.63)(float) | (118.20,39.92,29.54)(uchar)<br>(163.17,51.71,35.21)(float) | (206.52,47.57,36.21)(uchar)<br>(188.24,43.54,30.93)(float) |
| WarpAffine<br>(Interpolation_linear+Border_constant) | (122.45,39.17,79.32)(uchar)<br>(98.71,50.77,40.85)(float) | (105.00,84.94,86.24)(uchar)<br>(94.37,67.42,72.24)(float) | (153.45,67.84,64.88)(uchar)<br>(99.46,67.94,47.77)(float) |
| WarpAffine<br>(Interpolation_linear+Border_replicate) | (176.23,57.95,86.96)(uchar)<br>(169.20,56.32,38.23)(float) | (80.54,53.13,53.11)(uchar)<br>(72.77,58.22,63.10)(float) | (87.71,41.60,41.29)(uchar)<br>(94.62,49.70,40.31)(float) |
| WarpAffine<br>(Interpolation_linear+Border_transparent) | (116.16,51.15,66.50)(uchar)<br>(115.34,58.52,52.60)(float) | (137.30,76.58,123.93)(uchar)<br>(127.21,106.05,67.76)(float) | (149.23,56.21,74.64)(uchar)<br>(130.76,54.95,59.38)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_constant) | (857.96,472.67,618.08)(uchar)<br>(643.74,360.72,254.93)(float) | (225.06,156.35,148.33)(uchar)<br>(214.70,142.96,176.15)(float) | (393.35,195.58,234.78)(uchar)<br>(338.46,198.13,215.80)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_replicate) | (1037.22,529.82,568.16)(uchar)<br>(548.95,537.74,192.03)(float) | (198.07,126.18,123.14)(uchar)<br>(188.13,161.62,124.16)(float) | (267.83,159.13,151.49)(uchar)<br>(226.19,92.31,148.92)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_transparent) | (1099.20,540.88,754.99)(uchar)<br>(785.35,533.69,139.03)(float) | (246.21,151.03,232.75)(uchar)<br>(159.41,176.40,175.77)(float) | (429.14,336.95,508.27)(uchar)<br>(357.31,292.69,361.01)(float) |

#### 4.2. Image Size: 640*480

| function | PPLCV | OpenCV v4.4 | OpenCV v4.5.5 |
| :-------: | :-------: | :-------: | :-------: |
| Add | (645.29,212.70,159.53)(uchar)<br>(534.54,177.49,135.80)(float) | (425.84,140.75,105.46)(uchar)<br>(199.31,66.22,49.91)(float) | (583.65,179.14,135.90)(uchar)<br>(134.20,41.38,35.95)(float) |
| AddWeighted | (754.54,236.46,175.89)(uchar)<br>(317.56,98.28,82.74)(float) | (126.89,42.27,31.52)(uchar)<br>(168.91,56.39,42.67)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Subtract | (647.47,213.07,160.03)(uchar)<br>(543.78,180.28,135.53)(float) | (424.26,140.37,105.33)(uchar)<br>(199.22,66.17,49.96)(float) | (583.46,178.34,135.89)(uchar)<br>(135.40,47.20,36.01)(float) |
| Mul | (614.16,201.73,152.00)(uchar)<br>(544.94,179.87,135.59)(float) | (302.49,101.08,75.81)(uchar)<br>(199.10,66.09,49.97)(float) | (528.21,164.79,124.57)(uchar)<br>(135.46,47.32,35.96)(float) |
| Div | (780.13,251.16,192.64)(uchar)<br>(228.36,77.14,58.06)(float) | (76.92,25.59,19.20)(uchar)<br>(110.73,36.45,27.30)(float) | (149.89,49.99,37.49)(uchar)<br>(105.74,35.61,26.94)(float) |
| BGR2BGRA | (413.73)(uchar)<br>(143.703)(float) | (271.043)(uchar)<br>(222.974)(float) | (935.668)(uchar)<br>(253.089)(float) |
| BGRA2BGR | (441.351)(uchar)<br>(185.206)(float) | (292.948)(uchar)<br>(239.695)(float) | (883.901)(uchar)<br>(250.521)(float) |
| BGR2GRAY | (507.978)(uchar)<br>(465.021)(float) | (185.15)(uchar)<br>(118.458)(float) | (162.779)(uchar)<br>(313.198)(float) |
| BGRA2GRAY | (484.054)(uchar)<br>(404.309)(float) | (184.989)(uchar)<br>(115.667)(float) | (160.808)(uchar)<br>(272.214)(float) |
| GRAY2BGR | (583.82)(uchar)<br>(441.903)(float) | (234.219)(uchar)<br>(86.01)(float) | (1697.1)(uchar)<br>(498.409)(float) |
| GRAY2BGRA | (197.628)(uchar)<br>(66.982)(float) | (185.569)(uchar)<br>(66.187)(float) | (1556.07)(uchar)<br>(410.557)(float) |
| NV2BGR<br>(NV122RGB_MODE) | (111.184)(uchar)<br>(-)(float) | (57.552)(uchar)<br>(-)(float) | (105.114)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV122BGR_MODE) | (111.22)(uchar)<br>(-)(float) | (56.968)(uchar)<br>(-)(float) | (109.785)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212RGB_MODE) | (111.753)(uchar)<br>(-)(float) | (57.445)(uchar)<br>(-)(float) | (105.109)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212BGR_MODE) | (111.751)(uchar)<br>(-)(float) | (57.396)(uchar)<br>(-)(float) | (109.814)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2I420_MODE) | (169.819)(uchar)<br>(-)(float) | (114.868)(uchar)<br>(-)(float) | (202.303)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2I420_MODE) | (170.213)(uchar)<br>(-)(float) | (123.641)(uchar)<br>(-)(float) | (207.551)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2YV12_MODE) | (170.24)(uchar)<br>(-)(float) | (114.845)(uchar)<br>(-)(float) | (201.786)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2YV12_MODE) | (170.266)(uchar)<br>(-)(float) | (123.556)(uchar)<br>(-)(float) | (205.724)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202RGB_MODE) | (109.018)(uchar)<br>(-)(float) | (57.41)(uchar)<br>(-)(float) | (104.649)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202BGR_MODE) | (108.994)(uchar)<br>(-)(float) | (57.412)(uchar)<br>(-)(float) | (105.17)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122RGB_MODE) | (108.527)(uchar)<br>(-)(float) | (57.32)(uchar)<br>(-)(float) | (104.464)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122BGR_MODE) | (109.011)(uchar)<br>(-)(float) | (57.21)(uchar)<br>(-)(float) | (105.154)(uchar)<br>(-)(float) |
| CopyMakeborder<br>(BORDER_CONSTANT) | (1630.82,794.45,950.87)(uchar)<br>(726.08,294.80,232.19)(float) | (2289.14,934.88,765.13)(uchar)<br>(766.50,300.95,244.72)(float) | (2225.73,941.27,762.53)(uchar)<br>(771.41,299.20,239.97)(float) |
| CopyMakeborder<br>(BORDER_REPLICATE) | (1504.62,778.78,666.87)(uchar)<br>(728.46,292.05,226.19)(float) | (1552.74,775.18,725.59)(uchar)<br>(726.01,290.73,228.31)(float) | (1491.68,774.23,715.38)(uchar)<br>(713.38,287.46,224.56)(float) |
| CopyMakeborder<br>(BORDER_REFLECT) | (1498.18,778.54,664.70)(uchar)<br>(724.25,291.45,226.01)(float) | (1545.96,773.26,724.62)(uchar)<br>(722.97,289.83,227.67)(float) | (1493.02,773.89,714.17)(uchar)<br>(711.78,286.70,224.37)(float) |
| CopyMakeborder<br>(BORDER_REFLECT101) | (1498.36,780.14,665.58)(uchar)<br>(727.00,291.48,226.08)(float) | (1548.29,774.31,725.40)(uchar)<br>(724.96,290.24,227.60)(float) | (1483.72,773.57,715.91)(uchar)<br>(710.38,286.45,223.92)(float) |
| Dilate<br>(k3x3) | (444.86,155.04,117.33)(uchar)<br>(140.64,35.99,18.46)(float) | (105.65,34.36,25.86)(uchar)<br>(54.32,15.62,11.71)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Dilate<br>(k5x5) | (227.34,89.62,65.98)(uchar)<br>(81.88,16.89,10.76)(float) | (75.85,24.31,17.70)(uchar)<br>(36.56,10.50,7.42)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k3x3) | (432.07,174.61,135.96)(uchar)<br>(131.89,48.67,18.58)(float) | (105.30,33.84,25.54)(uchar)<br>(54.32,15.61,11.60)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k5x5) | (223.39,91.92,72.06)(uchar)<br>(70.58,23.74,10.74)(float) | (75.66,24.14,17.76)(uchar)<br>(35.87,10.43,7.63)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Flip<br>(mode:-1) | (1382.62,358.06,289.67)(uchar)<br>(235.63,158.39,137.41)(float) | (399.37,127.18,93.60)(uchar)<br>(93.49,30.67,23.26)(float) | (632.83,159.88,139.55)(uchar)<br>(139.77,60.77,36.80)(float) |
| Flip<br>(mode:0) | (2672.75,1743.99,1488.02)(uchar)<br>(951.05,487.85,405.17)(float) | (861.67,324.18,250.59)(uchar)<br>(250.31,85.42,65.11)(float) | (832.08,317.76,243.00)(uchar)<br>(240.28,78.29,59.48)(float) |
| Flip<br>(mode:1) | (1238.06,356.72,292.80)(uchar)<br>(234.15,162.11,139.31)(float) | (511.03,161.04,108.71)(uchar)<br>(108.75,35.12,28.40)(float) | (892.51,186.30,167.93)(uchar)<br>(167.60,74.82,45.87)(float) |
| Resize<br>(INTERPOLATION_NEAREST_LINEAR)<br>(640/480/320/240) | (3160.96,336.383,578.59)(uchar)<br>(755.676,328.05,77.415)(float) | (865.89,318.809,252.087)(uchar)<br>(193.023,50.93,38.31)(float) | (2002.27,203.01,567.041)(uchar)<br>(774.865,50.875,73.68)(float) |
| Resize<br>(INTERPOLATION_NEAREST_POINT)<br>(640/480/320/240) | (2949.32,779.33,989.628)(uchar)<br>(1049.17,646.455,491.99)(float) | (1876.47,809.70,889.35)(uchar)<br>(886.13,609.72,99.41)(float) | (1818.71,785.21,761.14)(uchar)<br>(761.46,558.664,68.27)(float) |
| Resize<br>(INTERPOLATION_AREA)<br>(640/480/320/240) | (2724.61,343.25,587.92)(uchar)<br>(759.29,327.54,77.71)(float) | (867.26,319.49,252.86)(uchar)<br>(195.77,50.95,38.35)(float) | (2002.84,203.03,567.35)(uchar)<br>(760.34,50.99,73.73)(float) |
| WarpAffine<br>(Interpolation_linear+Border_constant) | (18.68,17.27,20.07)(uchar)<br>(16.93,10.67,8.42)(float) | (31.54,22.23,16.76)(uchar)<br>(26.39,17.77,19.20)(float) | (32.22,13.68,18.06)(uchar)<br>(35.77,15.34,13.02)(float) |
| WarpAffine<br>(Interpolation_linear+Border_replicate) | (17.54,16.99,19.64)(uchar)<br>(17.06,8.00,7.33)(float) | (20.59,15.16,12.98)(uchar)<br>(20.28,12.42,12.64)(float) | (21.56,9.71,10.09)(uchar)<br>(20.89,15.97,11.98)(float) |
| WarpAffine<br>(Interpolation_linear+Border_transparent) | (30.72,17.28,15.72)(uchar)<br>(16.55,13.93,12.01)(float) | (36.50,21.72,18.45)(uchar)<br>(31.16,17.06,11.84)(float) | (39.98,19.31,29.34)(uchar)<br>(22.38,28.44,16.54)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_constant) | (174.53,101.74,116.32)(uchar)<br>(133.93,78.70,53.94)(float) | (56.88,42.71,23.10)(uchar)<br>(53.85,42.11,38.65)(float) | (95.72,58.99,55.95)(uchar)<br>(83.41,38.15,43.26)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_replicate) | (197.79,112.47,183.86)(uchar)<br>(69.82,27.49,77.70)(float) | (47.36,49.32,35.37)(uchar)<br>(34.99,35.46,21.73)(float) | (72.93,52.97,30.38)(uchar)<br>(68.35,26.09,28.11)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_transparent) | (242.07,75.20,140.35)(uchar)<br>(76.28,99.27,93.58)(float) | (61.64,57.92,50.82)(uchar)<br>(62.26,65.67,52.72)(float) | (98.00,104.37,55.20)(uchar)<br>(88.29,59.85,96.34)(float) |

#### 4.3. Image Size: 1280*960

| function | PPLCV | OpenCV v4.4 | OpenCV v4.5.5 |
| :-------: | :-------: | :-------: | :-------: |
| Add | (211.97,70.61,55.29)(uchar)<br>(174.13,59.42,44.91)(float) | (140.74,47.13,35.37)(uchar)<br>(66.20,22.28,16.58)(float) | (179.35,60.94,47.28)(uchar)<br>(42.87,12.79,11.81)(float) |
| AddWeighted | (238.70,74.23,55.88)(uchar)<br>(102.53,22.49,21.42)(float) | (42.27,14.11,10.58)(uchar)<br>(56.13,18.70,14.10)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Subtract | (212.65,71.28,54.42)(uchar)<br>(180.34,60.05,44.85)(float) | (140.40,46.91,35.23)(uchar)<br>(66.07,22.29,16.57)(float) | (178.86,60.93,47.36)(uchar)<br>(47.35,16.10,11.82)(float) |
| Mul | (201.27,68.22,51.09)(uchar)<br>(179.44,59.60,44.88)(float) | (101.10,33.79,25.33)(uchar)<br>(66.01,22.23,16.58)(float) | (164.82,55.95,43.31)(uchar)<br>(47.26,16.11,11.81)(float) |
| Div | (245.47,88.73,61.31)(uchar)<br>(75.38,25.56,19.18)(float) | (25.64,8.53,6.40)(uchar)<br>(36.42,12.10,9.09)(float) | (49.99,16.68,12.51)(uchar)<br>(35.62,12.02,8.90)(float) |
| BGR2BGRA | (126.65)(uchar)<br>(43.82)(float) | (91.725)(uchar)<br>(78.57)(float) | (327.04)(uchar)<br>(87.893)(float) |
| BGRA2BGR | (147.741)(uchar)<br>(74.096)(float) | (99.677)(uchar)<br>(82.23)(float) | (312.884)(uchar)<br>(85.711)(float) |
| BGR2GRAY | (168.788)(uchar)<br>(155.211)(float) | (62.358)(uchar)<br>(39.582)(float) | (55.022)(uchar)<br>(106.61)(float) |
| BGRA2GRAY | (161.06)(uchar)<br>(136.192)(float) | (61.99)(uchar)<br>(38.652)(float) | (54.572)(uchar)<br>(92.276)(float) |
| GRAY2BGR | (194.401)(uchar)<br>(146.89)(float) | (78.245)(uchar)<br>(28.811)(float) | (583.923)(uchar)<br>(171.158)(float) |
| GRAY2BGRA | (65.928)(uchar)<br>(22.20)(float) | (61.989)(uchar)<br>(22.23)(float) | (529.824)(uchar)<br>(139.76)(float) |
| NV2BGR<br>(NV122RGB_MODE) | (37.071)(uchar)<br>(-)(float) | (18.787)(uchar)<br>(-)(float) | (35.009)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV122BGR_MODE) | (37.094)(uchar)<br>(-)(float) | (18.708)(uchar)<br>(-)(float) | (36.692)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212RGB_MODE) | (37.243)(uchar)<br>(-)(float) | (18.764)(uchar)<br>(-)(float) | (35.043)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212BGR_MODE) | (37.202)(uchar)<br>(-)(float) | (18.713)(uchar)<br>(-)(float) | (36.692)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2I420_MODE) | (56.469)(uchar)<br>(-)(float) | (38.643)(uchar)<br>(-)(float) | (68.536)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2I420_MODE) | (56.677)(uchar)<br>(-)(float) | (41.666)(uchar)<br>(-)(float) | (69.922)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2YV12_MODE) | (56.629)(uchar)<br>(-)(float) | (38.749)(uchar)<br>(-)(float) | (67.988)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2YV12_MODE) | (56.662)(uchar)<br>(-)(float) | (41.587)(uchar)<br>(-)(float) | (69.506)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202RGB_MODE) | (36.424)(uchar)<br>(-)(float) | (18.832)(uchar)<br>(-)(float) | (34.933)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202BGR_MODE) | (36.414)(uchar)<br>(-)(float) | (18.699)(uchar)<br>(-)(float) | (35.046)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122RGB_MODE) | (36.432)(uchar)<br>(-)(float) | (18.83)(uchar)<br>(-)(float) | (34.945)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122BGR_MODE) | (36.429)(uchar)<br>(-)(float) | (18.796)(uchar)<br>(-)(float) | (35.071)(uchar)<br>(-)(float) |
| CopyMakeborder<br>(BORDER_CONSTANT) | (733.49,336.80,335.50)(uchar)<br>(281.95,107.61,83.31)(float) | (905.09,367.86,296.22)(uchar)<br>(296.37,108.21,85.43)(float) | (922.96,375.51,296.00)(uchar)<br>(300.96,107.42,83.56)(float) |
| CopyMakeborder<br>(BORDER_REPLICATE) | (677.98,333.68,272.02)(uchar)<br>(286.48,106.27,81.91)(float) | (698.64,332.43,287.14)(uchar)<br>(287.11,106.46,82.34)(float) | (704.47,332.63,283.52)(uchar)<br>(282.44,105.93,81.37)(float) |
| CopyMakeborder<br>(BORDER_REFLECT) | (677.15,332.22,271.25)(uchar)<br>(286.23,106.07,81.97)(float) | (698.71,331.91,286.30)(uchar)<br>(285.88,106.27,82.27)(float) | (705.67,332.06,282.82)(uchar)<br>(280.90,105.87,81.14)(float) |
| CopyMakeborder<br>(BORDER_REFLECT101) | (677.54,333.38,271.32)(uchar)<br>(286.14,106.11,81.62)(float) | (698.35,331.75,286.82)(uchar)<br>(286.62,106.29,82.07)(float) | (703.15,332.30,283.65)(uchar)<br>(282.83,105.51,81.23)(float) |
| Dilate<br>(k3x3) | (153.22,54.10,38.84)(uchar)<br>(45.15,8.28,6.39)(float) | (35.36,11.09,8.18)(uchar)<br>(17.43,5.31,2.83)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Dilate<br>(k5x5) | (766.2550,317.7950,248.1010)(uchar)<br>(302.9960,86.7132,55.3967)(float) | (297.5410,99.6072,73.8433)(uchar)<br>(154.1220,43.8696,30.8186)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k3x3) | (166.01,61.81,47.26)(uchar)<br>(44.63,12.69,6.20)(float) | (35.18,10.93,8.32)(uchar)<br>(18.30,5.14,2.93)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k5x5) | (87.79,30.74,19.34)(uchar)<br>(18.75,7.96,3.41)(float) | (24.62,7.57,5.47)(uchar)<br>(11.47,3.44,2.10)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Flip<br>(mode:-1) | (459.48,119.14,97.73)(uchar)<br>(78.42,52.22,37.13)(float) | (132.45,40.69,30.07)(uchar)<br>(30.10,9.75,7.68)(float) | (207.32,56.70,45.22)(uchar)<br>(45.45,18.72,11.98)(float) |
| Flip<br>(mode:0) | (1097.36,625.66,533.58)(uchar)<br>(330.90,165.72,136.90)(float) | (305.87,112.65,77.36)(uchar)<br>(77.32,25.80,22.13)(float) | (296.09,105.11,72.39)(uchar)<br>(72.74,23.88,19.78)(float) |
| Flip<br>(mode:1) | (439.37,117.48,97.64)(uchar)<br>(77.73,51.04,23.33)(float) | (171.65,50.12,37.83)(uchar)<br>(37.85,11.53,9.46)(float) | (277.21,64.74,57.90)(uchar)<br>(58.22,25.38,14.87)(float) |
| Resize<br>(INTERPOLATION_NEAREST_LINEAR)<br>(1280/960/800/600) | (112.69,21.25,32.85)(uchar)<br>(61.03,17.59,15.62)(float) | (51.24,17.43,11.33)(uchar)<br>(45.18,13.26,9.58)(float) | (64.74,11.00,10.50)(uchar)<br>(41.59,11.17,8.69)(float) |
| Resize<br>(INTERPOLATION_NEAREST_POINT)<br>(1280/960/800/600) | (267.97,86.93,108.84)(uchar)<br>(101.15,35.85,27.39)(float) | (206.81,81.81,81.93)(uchar)<br>(81.92,34.81,20.61)(float) | (209.05,83.33,81.65)(uchar)<br>(81.63,34.87,12.01)(float) |
| Resize<br>(INTERPOLATION_AREA)<br>(1280/960/800/600) | (21.18,11.11,8.32)(uchar)<br>(18.57,7.62,5.16)(float) | (19.49,7.80,5.83)(uchar)<br>(17.84,6.25,4.62)(float) | (18.18,7.36,5.49)(uchar)<br>(18.28,6.91,5.09)(float) |
| WarpAffine<br>(Interpolation_linear+Border_constant) | (6.71,5.51,7.36)(uchar)<br>(6.56,2.93,3.23)(float) | (6.34,6.94,6.99)(uchar)<br>(9.53,4.99,4.57)(float) | (13.91,7.30,5.90)(uchar)<br>(7.39,7.66,6.20)(float) |
| WarpAffine<br>(Interpolation_linear+Border_replicate) | (6.02,6.65,5.79)(uchar)<br>(9.53,3.90,3.82)(float) | (7.13,5.30,3.65)(uchar)<br>(6.20,3.92,3.50)(float) | (8.44,3.94,3.21)(uchar)<br>(4.88,3.83,3.48)(float) |
| WarpAffine<br>(Interpolation_linear+Border_transparent) | (5.21,5.83,3.20)(uchar)<br>(6.56,3.28,4.43)(float) | (12.50,6.37,8.97)(uchar)<br>(9.85,6.21,7.84)(float) | (13.80,4.65,8.53)(uchar)<br>(13.39,6.14,5.20)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_constant) | (61.63,39.15,43.36)(uchar)<br>(51.82,29.83,18.12)(float) | (20.01,12.94,15.15)(uchar)<br>(17.92,11.84,14.41)(float) | (30.10,19.98,18.51)(uchar)<br>(24.68,18.37,16.71)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_replicate) | (47.49,35.31,38.97)(uchar)<br>(36.33,23.94,28.91)(float) | (16.38,11.62,10.99)(uchar)<br>(13.40,11.54,9.37)(float) | (17.12,15.88,11.16)(uchar)<br>(11.69,8.22,8.13)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_transparent) | (53.66,45.91,58.50)(uchar)<br>(57.63,33.38,29.99)(float) | (21.40,23.35,19.11)(uchar)<br>(20.97,16.51,17.55)(float) | (40.18,34.13,42.46)(uchar)<br>(31.48,16.63,15.94)(float) |

#### 4.4. Image Size: 1920*1080

| function | PPLCV | OpenCV v4.4 | OpenCV v4.5.5 |
| :-------: | :-------: | :-------: | :-------: |
| Add | (94.62,31.60,23.85)(uchar)<br>(73.96,21.33,14.60)(float) | (62.94,21.02,15.73)(uchar)<br>(29.63,8.03,5.45)(float) | (88.87,29.51,18.44)(uchar)<br>(15.56,5.42,3.78)(float) |
| AddWeighted | (102.67,33.23,25.69)(uchar)<br>(25.36,6.79,4.76)(float) | (18.81,6.27,4.70)(uchar)<br>(25.36,6.79,4.76)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Subtract | (95.48,31.99,23.38)(uchar)<br>(79.79,21.32,14.56)(float) | (62.65,20.90,15.64)(uchar)<br>(29.67,8.02,5.45)(float) | (88.81,29.67,18.36)(uchar)<br>(18.30,5.43,3.80)(float) |
| Mul | (90.01,30.09,22.46)(uchar)<br>(80.59,21.34,14.58)(float) | (45.01,14.99,11.27)(uchar)<br>(29.65,8.04,5.46)(float) | (79.46,26.54,17.10)(uchar)<br>(18.36,5.42,3.79)(float) |
| Div | (115.84,40.68,28.94)(uchar)<br>(31.84,9.43,6.53)(float) | (11.40,3.80,2.85)(uchar)<br>(16.76,4.59,3.14)(float) | (22.15,7.41,5.56)(uchar)<br>(15.32,4.33,3.09)(float) |
| BGR2BGRA | (52.297)(uchar)<br>(18.11)(float) | (41.294)(uchar)<br>(35.781)(float) | (150.493)(uchar)<br>(40.193)(float) |
| BGRA2BGR | (66.578)(uchar)<br>(35.22)(float) | (44.91)(uchar)<br>(36.952)(float) | (143.818)(uchar)<br>(38.304)(float) |
| BGR2GRAY | (74.971)(uchar)<br>(69.021)(float) | (27.757)(uchar)<br>(16.663)(float) | (24.558)(uchar)<br>(47.866)(float) |
| BGRA2GRAY | (71.663)(uchar)<br>(60.452)(float) | (27.592)(uchar)<br>(17.186)(float) | (24.350)(uchar)<br>(47.115)(float) |
| GRAY2BGR | (86.29)(uchar)<br>(65.431)(float) | (35.017)(uchar)<br>(12.838)(float) | (273.449)(uchar)<br>(76.592)(float) |
| GRAY2BGRA | (29.324)(uchar)<br>(7.359)(float) | (27.581)(uchar)<br>(9.89)(float) | (239.474)(uchar)<br>(62.568)(float) |
| NV2BGR<br>(NV122RGB_MODE) | (16.483)(uchar)<br>(-)(float) | (8.42604)(uchar)<br>(-)(float) | (15.5878)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV122BGR_MODE) | (16.489)(uchar)<br>(-)(float) | (8.4138)(uchar)<br>(-)(float) | (16.2555)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212RGB_MODE) | (16.5559)(uchar)<br>(-)(float) | (8.44105)(uchar)<br>(-)(float) | (15.5793)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212BGR_MODE) | (16.5138)(uchar)<br>(-)(float) | (8.42671)(uchar)<br>(-)(float) | (16.3394)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2I420_MODE) | (25.1301)(uchar)<br>(-)(float) | (17.2989)(uchar)<br>(-)(float) | (30.6456)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2I420_MODE) | (25.0519)(uchar)<br>(-)(float) | (18.6546)(uchar)<br>(-)(float) | (31.2895)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2YV12_MODE) | (25.1402)(uchar)<br>(-)(float) | (17.2937)(uchar)<br>(-)(float) | (30.4463)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2YV12_MODE) | (25.1412)(uchar)<br>(-)(float) | (18.6355)(uchar)<br>(-)(float) | (31.0771)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202RGB_MODE) | (16.135)(uchar)<br>(-)(float) | (8.40128)(uchar)<br>(-)(float) | (15.555)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202BGR_MODE) | (16.1823)(uchar)<br>(-)(float) | (8.37281)(uchar)<br>(-)(float) | (15.5988)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122RGB_MODE) | (16.2143)(uchar)<br>(-)(float) | (8.38921)(uchar)<br>(-)(float) | (15.5551)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122BGR_MODE) | (16.2182)(uchar)<br>(-)(float) | (8.37594)(uchar)<br>(-)(float) | (15.5849)(uchar)<br>(-)(float) |
| CopyMakeborder<br>(BORDER_CONSTANT) | (385.61,165.57,151.49)(uchar)<br>(127.97,49.63,38.03)(float) | (453.04,176.73,139.71)(uchar)<br>(140.31,49.67,38.65)(float) | (461.59,178.99,140.12)(uchar)<br>(140.79,41.52,29.16)(float) |
| CopyMakeborder<br>(BORDER_REPLICATE) | (362.05,164.34,131.43)(uchar)<br>(136.21,49.23,37.20)(float) | (370.93,163.61,137.09)(uchar)<br>(137.06,49.36,37.50)(float) | (373.84,163.89,135.93)(uchar)<br>(135.15,41.06,28.69)(float) |
| CopyMakeborder<br>(BORDER_REFLECT) | (362.92,163.50,130.32)(uchar)<br>(136.34,49.09,37.15)(float) | (370.54,163.24,136.89)(uchar)<br>(136.93,49.15,37.48)(float) | (373.06,163.65,135.64)(uchar)<br>(135.27,41.21,28.69)(float) |
| CopyMakeborder<br>(BORDER_REFLECT101) | (362.15,164.18,130.94)(uchar)<br>(136.16,49.16,37.15)(float) | (369.57,163.42,136.87)(uchar)<br>(136.93,49.34,37.47)(float) | (373.49,163.60,135.95)(uchar)<br>(135.36,41.17,28.70)(float) |
| Dilate<br>(k3x3) | (69.15,23.37,17.18)(uchar)<br>(18.23,3.53,2.15)(float) | (15.56,4.92,3.68)(uchar)<br>(7.56,1.54,1.10)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Dilate<br>(k5x5) | (39.03,12.54,8.98)(uchar)<br>(10.75,1.64,1.39)(float) | (10.93,3.30,2.42)(uchar)<br>(5.04,1.15,0.69)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k3x3) | (77.99,22.55,18.13)(uchar)<br>(16.64,5.23,2.17)(float) | (15.40,4.89,3.63)(uchar)<br>(8.03,1.53,1.12)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k5x5) | (41.41,11.55,6.57)(uchar)<br>(6.37,3.46,1.36)(float) | (10.85,3.28,2.40)(uchar)<br>(4.96,1.14,0.83)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Flip<br>(mode:-1) | (200.59,52.97,43.27)(uchar)<br>(34.85,17.54,8.07)(float) | (59.98,18.15,13.65)(uchar)<br>(13.67,4.51,2.76)(float) | (82.06,24.67,22.05)(uchar)<br>(21.46,8.76,4.44)(float) |
| Flip<br>(mode:0) | (560.32,289.24,233.60)(uchar)<br>(149.00,73.83,55.65)(float) | (145.33,50.02,37.86)(uchar)<br>(37.82,12.80,7.60)(float) | (122.32,45.24,34.61)(uchar)<br>(34.84,11.60,7.26)(float) |
| Flip<br>(mode:1) | (198.78,52.67,43.19)(uchar)<br>(34.69,11.52,8.09)(float) | (76.73,22.72,16.90)(uchar)<br>(16.91,5.14,3.17)(float) | (111.73,28.22,26.47)(uchar)<br>(25.86,11.28,5.18)(float) |
| WarpAffine<br>(Interpolation_linear+Border_constant) | (3.62,2.15,2.75)(uchar)<br>(2.11,1.04,1.22)(float) | (4.22,3.33,3.04)(uchar)<br>(4.00,1.92,1.74)(float) | (4.80,2.67,1.85)(uchar)<br>(3.72,1.93,1.58)(float) |
| WarpAffine<br>(Interpolation_linear+Border_replicate) | (4.22,1.84,1.97)(uchar)<br>(3.97,1.24,1.35)(float) | (3.24,2.41,1.80)(uchar)<br>(2.81,1.39,1.49)(float) | (3.55,1.72,1.68)(uchar)<br>(3.05,1.50,1.27)(float) |
| WarpAffine<br>(Interpolation_linear+Border_transparent) | (2.61,1.95,1.97)(uchar)<br>(2.73,1.29,1.06)(float) | (5.50,2.95,3.88)(uchar)<br>(3.83,2.48,3.61)(float) | (6.49,3.27,3.45)(uchar)<br>(3.78,2.84,2.96)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_constant) | (18.50,19.92,20.40)(uchar)<br>(18.90,11.00,6.71)(float) | (8.68,6.34,5.57)(uchar)<br>(7.13,5.90,4.30)(float) | (13.68,10.44,7.08)(uchar)<br>(11.27,7.86,5.90)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_replicate) | (30.06,13.14,17.44)(uchar)<br>(10.69,9.41,6.38)(float) | (7.34,4.67,4.70)(uchar)<br>(7.97,4.99,3.49)(float) | (12.91,6.04,7.08)(uchar)<br>(9.63,4.49,3.04)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_transparent) | (19.59,17.01,21.56)(uchar)<br>(14.94,19.49,9.75)(float) | (8.81,10.02,10.06)(uchar)<br>(7.97,6.47,6.66)(float) | (17.22,14.63,16.40)(uchar)<br>(11.68,8.37,13.72)(float) |

#### 4.5. Image Size: 3840*2160

| function | PPLCV | OpenCV v4.4 | OpenCV v4.5.5 |
| :-------: | :-------: | :-------: | :-------: |
| Add | (24.14,6.55,4.64)(uchar)<br>(15.43,1.65,1.23)(float) | (15.73,4.38,3.04)(uchar)<br>(5.44,1.10,0.83)(float) | (21.57,5.41,3.76)(uchar)<br>(3.83,0.99,0.75)(float) |
| AddWeighted | (25.06,6.72,5.06)(uchar)<br>(7.61,1.70,1.28)(float) | (4.71,1.32,0.99)(uchar)<br>(4.69,1.05,0.79)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Subtract | (23.94,6.53,4.68)(uchar)<br>(14.55,1.65,1.24)(float) | (15.61,4.35,3.03)(uchar)<br>(5.44,1.10,0.83)(float) | (21.58,5.40,3.76)(uchar)<br>(3.77,1.00,0.75)(float) |
| Mul | (23.28,5.94,4.44)(uchar)<br>(14.58,1.66,1.24)(float) | (11.28,3.10,1.99)(uchar)<br>(5.45,1.10,0.83)(float) | (19.55,5.06,3.53)(uchar)<br>(3.77,1.00,0.75)(float) |
| Div | (28.13,7.33,5.11)(uchar)<br>(6.55,1.29,0.97)(float) | (2.85,0.85,0.64)(uchar)<br>(3.14,0.83,0.62)(float) | (5.56,1.53,1.15)(uchar)<br>(3.08,0.85,0.64)(float) |
| BGR2BGRA | (13.5309)(uchar)<br>(1.18028)(float) | (10.402)(uchar)<br>(1.1787)(float) | (38.3996)(uchar)<br>(1.17148)(float) |
| BGRA2BGR | (16.7388)(uchar)<br>(1.12305)(float) | (11.3216)(uchar)<br>(1.28357)(float) | (37.2489)(uchar)<br>(1.27721)(float) |
| BGR2GRAY | (18.75)(uchar)<br>(9.46362)(float) | (6.94957)(uchar)<br>(4.37089)(float) | (6.16552)(uchar)<br>(12.1099)(float) |
| BGRA2GRAY | (17.8368)(uchar)<br>(7.23298)(float) | (6.91245)(uchar)<br>(4.28602)(float) | (6.1095)(uchar)<br>(10.4432)(float) |
| GRAY2BGR | (21.6445)(uchar)<br>(3.80429)(float) | (8.82131)(uchar)<br>(1.3973)(float) | (71.0489)(uchar)<br>(5.49375)(float) |
| GRAY2BGRA | (7.32073)(uchar)<br>(0.784571)(float) | (6.90969)(uchar)<br>(1.06463)(float) | (61.4536)(uchar)<br>(1.68015)(float) |
| NV2BGR<br>(NV122RGB_MODE) | (4.12104)(uchar)<br>(-)(float) | (2.10888)(uchar)<br>(-)(float) | (3.89315)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV122BGR_MODE) | (4.12374)(uchar)<br>(-)(float) | (2.09964)(uchar)<br>(-)(float) | (4.08625)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212RGB_MODE) | (4.13566)(uchar)<br>(-)(float) | (2.10959)(uchar)<br>(-)(float) | (3.87174)(uchar)<br>(-)(float) |
| NV2BGR<br>(NV212BGR_MODE) | (4.14236)(uchar)<br>(-)(float) | (2.10226)(uchar)<br>(-)(float) | (4.08722)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2I420_MODE) | (6.28701)(uchar)<br>(-)(float) | (4.31895)(uchar)<br>(-)(float) | (7.69141)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2I420_MODE) | (6.28734)(uchar)<br>(-)(float) | (4.68189)(uchar)<br>(-)(float) | (7.83363)(uchar)<br>(-)(float) |
| Color2YUV420<br>(RGB2YV12_MODE) | (6.29212)(uchar)<br>(-)(float) | (4.33935)(uchar)<br>(-)(float) | (7.63036)(uchar)<br>(-)(float) |
| Color2YUV420<br>(BGR2YV12_MODE) | (6.28365)(uchar)<br>(-)(float) | (4.6706)(uchar)<br>(-)(float) | (7.79329)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202RGB_MODE) | (4.06256)(uchar)<br>(-)(float) | (2.10925)(uchar)<br>(-)(float) | (3.88084)(uchar)<br>(-)(float) |
| YUV4202Color<br>(I4202BGR_MODE) | (4.06315)(uchar)<br>(-)(float) | (2.10801)(uchar)<br>(-)(float) | (3.89592)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122RGB_MODE) | (4.0625)(uchar)<br>(-)(float) | (2.11273)(uchar)<br>(-)(float) | (3.88538)(uchar)<br>(-)(float) |
| YUV4202Color<br>(YV122BGR_MODE) | (4.06143)(uchar)<br>(-)(float) | (2.10696)(uchar)<br>(-)(float) | (3.8962)(uchar)<br>(-)(float) |
| CopyMakeborder<br>(BORDER_CONSTANT) | (120.06,46.70,38.81)(uchar)<br>(28.67,2.00,1.50)(float) | (131.70,48.19,36.93)(uchar)<br>(37.08,2.00,1.49)(float) | (132.65,40.21,28.72)(uchar)<br>(28.78,1.99,1.49)(float) |
| CopyMakeborder<br>(BORDER_REPLICATE) | (114.76,46.59,36.05)(uchar)<br>(36.74,1.99,1.48)(float) | (115.96,46.31,36.93)(uchar)<br>(36.93,1.99,1.49)(float) | (116.91,38.59,28.14)(uchar)<br>(28.08,1.98,1.49)(float) |
| CopyMakeborder<br>(BORDER_REFLECT) | (114.59,46.35,35.99)(uchar)<br>(36.68,1.99,1.49)(float) | (116.21,46.30,36.91)(uchar)<br>(36.92,1.98,1.49)(float) | (116.85,38.63,28.13)(uchar)<br>(28.07,1.99,1.49)(float) |
| CopyMakeborder<br>(BORDER_REFLECT101) | (114.59,46.41,35.97)(uchar)<br>(36.70,1.99,1.48)(float) | (115.91,46.31,36.93)(uchar)<br>(36.93,1.98,1.49)(float) | (116.29,38.58,27.99)(uchar)<br>(28.05,1.99,1.49)(float) |
| Dilate<br>(k3x3) | (16.87,3.79,2.38)(uchar)<br>(1.96,0.61,0.46)(float) | (3.90,1.06,0.86)(uchar)<br>(1.55,0.38,0.26)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Dilate<br>(k5x5) | (10.18,1.99,1.56)(uchar)<br>(1.92,0.41,0.31)(float) | (2.62,0.74,0.56)(uchar)<br>(1.03,0.29,0.16)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k3x3) | (20.62,5.05,2.92)(uchar)<br>(1.95,0.61,0.46)(float) | (3.92,1.06,0.85)(uchar)<br>(1.55,0.38,0.26)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Erode<br>(k5x5) | (10.26,1.68,1.43)(uchar)<br>(1.26,0.41,0.31)(float) | (2.62,0.74,0.56)(uchar)<br>(1.02,0.29,0.20)(float) | (-,-,-)(uchar)<br>(-,-,-)(float) |
| Flip<br>(mode:-1) | (48.46,12.63,9.42)(uchar)<br>(6.81,2.35,1.61)(float) | (14.40,3.66,2.64)(uchar)<br>(2.65,0.74,0.57)(float) | (19.74,5.32,3.78)(uchar)<br>(19.74,5.32,3.78)(float) |
| Flip<br>(mode:0) | (137.71,69.44,56.35)(uchar)<br>(30.30,11.16,7.64)(float) | (35.52,10.68,6.61)(uchar)<br>(6.60,1.32,1.02)(float) | (32.10,9.72,5.83)(uchar)<br>(5.92,1.31,1.00)(float) |
| Flip<br>(mode:1) | (48.10,12.34,7.91)(uchar)<br>(6.71,2.37,1.64)(float) | (18.24,4.64,3.16)(uchar)<br>(3.17,0.79,0.59)(float) | (24.36,5.95,5.00)(uchar)<br>(5.00,1.29,0.89)(float) |
| WarpAffine<br>(Interpolation_linear+Border_constant) | (0.75,0.48,0.50)(uchar)<br>(0.60,0.34,0.22)(float) | (1.05,0.64,0.72)(uchar)<br>(0.78,0.44,0.48)(float) | (1.12,0.51,0.46)(uchar)<br>(0.90,0.46,0.49)(float) |
| WarpAffine<br>(Interpolation_linear+Border_replicate) | (0.72,0.53,0.66)(uchar)<br>(0.63,0.34,0.18)(float) | (0.77,0.54,0.37)(uchar)<br>(0.61,0.36,0.32)(float) | (0.92,0.42,0.40)(uchar)<br>(0.69,0.34,0.31)(float) |
| WarpAffine<br>(Interpolation_linear+Border_transparent) | (0.65,0.46,0.44)(uchar)<br>(0.55,0.26,0.30)(float) | (1.41,0.74,1.09)(uchar)<br>(0.78,0.53,0.38)(float) | (1.60,0.66,0.56)(uchar)<br>(0.58,0.60,0.54)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_constant) | (6.59,2.52,1.28)(uchar)<br>(1.44,0.98,1.06)(float) | (2.12,1.23,1.20)(uchar)<br>(1.50,0.85,0.75)(float) | (3.54,1.69,1.51)(uchar)<br>(1.81,0.97,0.85)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_replicate) | (7.66,3.00,2.07)(uchar)<br>(3.23,1.20,1.16)(float) | (1.68,1.07,0.97)(uchar)<br>(1.24,0.84,0.74)(float) | (1.36,1.24,1.29)(uchar)<br>(1.96,0.79,0.64)(float) |
| WarpAffine<br>(Interpolation_nearest_point+Border_transparent) | (8.52,4.53,6.51)(uchar)<br>(2.18,3.16,1.30)(float) | (2.11,1.67,1.61)(uchar)<br>(0.95,1.25,1.64)(float) | (4.40,3.83,1.06)(uchar)<br>(2.50,1.74,1.75)(float) |