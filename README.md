# cordvox
NSF HiFi-GANの学習手法の改善

## 改善点
- 損失関数: メルスペクトログラム損失を multi-scale STFT Lossに変更
- DiscriminatorをMSDのみにする。
- 2ステージ学習による高効率な学習: 学習初期はDiscriminatorを使用しないことで計算効率を向上。