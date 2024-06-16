# cordvox : Modified NSF HiFi-GAN

## HiFi-GANとの相違点
### 励起信号発振器
F0と同じ周波数の正弦波を生成するオシレータを搭載。ソースフィルタモデルとする。
ノイズを加算することにより非周期成分の生成も可能。

### 損失関数
メルスペクトル損失をMultiscale STFT Lossに置き換え。高周波成分の解像度を高くする。

### MultiResolutionalDiscriminator
UnivNetに採用されている振幅スペクトログラムを二次元畳み込みするDiscriminator。

### 活性化関数
EVA-GAN(https://arxiv.org/abs/2402.00892)を参考に
LeakyReLUはなめらかではないのでより滑らかなSiLUに変更。

### Cyclic Noise
Cyclic Noiseを生成するオシレータを試験的に実装。
試してみたがうまく周期成分を抽出できず、HiFi-GAN側のアップサンプリング層で周期成分を生成することになってしまう。
https://www.isca-archive.org/interspeech_2020/wang20u_interspeech.pdf

### SAN
SAN損失を追加。
https://arxiv.org/abs/2301.12811