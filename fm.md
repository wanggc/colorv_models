```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("./input/uc.csv")
train_data = train_data.sample(frac=1.0)

from sklearn.preprocessing import LabelEncoder

ule = LabelEncoder()
vle = LabelEncoder()

train_data.userId = ule.fit_transform(train_data.userId)
train_data.videoId = vle.fit_transform(train_data.videoId)

print(train_data.videoId.max())
print(train_data.userId.max())
print(train_data.shape)
```

    9151
    35640
    (5928943, 3)



```python
from keras.layers import * #Input, Embedding, Dense,Flatten, merge,Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
import itertools
import keras
from keras.optimizers import *
from keras.regularizers import l2

from keras.utils import plot_model


def KerasFM(max_features,K=8,solver=Adam(lr=0.01),l2=0.00,l2_fm = 0.00):
    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    fm_layers = []
    #for c in columns:
    for c in max_features.keys():
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%c,
                        embeddings_regularizer=keras.regularizers.l2(1e-5)
                        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)
        
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = dot([emb1,emb2],axes=-1,normalize=True)
        fm_layers.append(dot_layer)

    #flatten = BatchNormalization(axis=1)(add((fm_layers)))
    flatten = dot_layer
    outputs = Dense(1,activation='sigmoid',name='outputs')(flatten)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=solver,loss= 'binary_crossentropy')
    plot_model(model, to_file='fm_cosine_model.png',show_shapes=True)
    #model.summary()
    return model
```

    Using TensorFlow backend.


    WARNING:tensorflow:From /root/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.



```python
cat_cols = ['userId', 'videoId']

max_features = train_data[cat_cols].max() + 1

train_len = int(len(train_data)*0.95)
X_train, X_valid = train_data[cat_cols][:train_len], train_data[cat_cols][train_len:]
y_train, y_valid = train_data['click'][:train_len], train_data['click'][train_len:]

train_input = []
valid_input = []

#print(test_data)
for col in cat_cols:
    train_input.append(X_train[col])
    valid_input.append(X_valid[col])
    
ck = keras.callbacks.ModelCheckpoint("best.model", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

model = KerasFM(max_features)
model.fit(train_input, y_train, batch_size=100000,nb_epoch=100,verbose=2,validation_data=(valid_input,y_valid),callbacks=[ck, es])
```

    /root/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:41: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=Tensor("ou...)`
    /root/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:21: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.


    WARNING:tensorflow:From /root/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 5632495 samples, validate on 296448 samples
    Epoch 1/100
     - 5s - loss: 0.5755 - val_loss: 0.4391
    Epoch 2/100
     - 5s - loss: 0.3879 - val_loss: 0.3548
    Epoch 3/100
     - 5s - loss: 0.3423 - val_loss: 0.3329
    Epoch 4/100
     - 5s - loss: 0.3272 - val_loss: 0.3233
    Epoch 5/100
     - 5s - loss: 0.3192 - val_loss: 0.3179
    Epoch 6/100
     - 5s - loss: 0.3136 - val_loss: 0.3137
    Epoch 7/100
     - 5s - loss: 0.3093 - val_loss: 0.3113
    Epoch 8/100
     - 5s - loss: 0.3060 - val_loss: 0.3091
    Epoch 9/100
     - 5s - loss: 0.3034 - val_loss: 0.3076
    Epoch 10/100
     - 5s - loss: 0.3011 - val_loss: 0.3065
    Epoch 11/100
     - 5s - loss: 0.2992 - val_loss: 0.3058
    Epoch 12/100
     - 5s - loss: 0.2976 - val_loss: 0.3049
    Epoch 13/100
     - 5s - loss: 0.2962 - val_loss: 0.3047
    Epoch 14/100
     - 5s - loss: 0.2950 - val_loss: 0.3043
    Epoch 15/100
     - 5s - loss: 0.2941 - val_loss: 0.3046
    Epoch 16/100
     - 5s - loss: 0.2931 - val_loss: 0.3038
    Epoch 17/100
     - 5s - loss: 0.2923 - val_loss: 0.3041
    Epoch 18/100
     - 5s - loss: 0.2916 - val_loss: 0.3040





    <keras.callbacks.History at 0x7f5cf4cde278>




```python


from sklearn.metrics import *

p_valid = model.predict(valid_input)
auc = roc_auc_score(y_valid, p_valid)
print("valid auc is %0.6f" % auc)

```

    valid auc is 0.864703



```python
import faiss
from faiss import normalize_L2


movie_emb_layer = model.get_layer('embed_videoId')
user_emb_layer = model.get_layer('embed_userId')   

(w_movie, ) = movie_emb_layer.get_weights()
(w_user, ) = user_emb_layer.get_weights()

normalize_L2(w_movie)
normalize_L2(w_user)
```


```python
search_vec = np.array(w_movie, dtype=np.float32)
index_vec = np.array(w_movie, dtype=np.float32)

index = faiss.IndexIDMap(faiss.IndexFlatIP(8))
index.add_with_ids(search_vec, vle.classes_)
D, I = index.search(index_vec, 10)
```


```python
I
```




    array([[ 85146166, 127260121, 127434064, ..., 127129423, 127153013,
            125883613],
           [ 96738179, 126931903, 127482928, ..., 127329366, 126745944,
            127237986],
           [100163377, 127106299, 127310466, ..., 126873861, 126968345,
            126816156],
           ...,
           [127578510, 124859461, 127182576, ..., 124940460, 127137066,
            127167119],
           [127591552, 125459033, 124390839, ..., 124035535, 126821267,
            127060990],
           [127596228, 126455547, 125655790, ..., 127087121, 127159037,
            124168358]])




```python
df = pd.DataFrame(I)
df.to_csv('videos.csv')
```
