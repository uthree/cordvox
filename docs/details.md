# cordvox : Unified Source-Filter Neural Vocoder

## HiFi-GANとの相違点
### 励起信号発振器
F0と同じ周波数の正弦波を生成するオシレータを搭載。ソースフィルタモデルとする。
ノイズを加算することにより非周期成分の生成も可能。

### 損失関数
DDSPのと同じMultiscale STFT Loss。
学習前半はDiscriminator無しで訓練すると短時間で学習できる。

### MultiResolutionalDiscriminator
UnivNetに採用されている振幅スペクトログラムを二次元畳み込みするDiscriminator。
