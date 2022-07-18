# 参考：https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
# モデルのコンパイル

from keras import optimizers

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])