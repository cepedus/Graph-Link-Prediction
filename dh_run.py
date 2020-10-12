import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import average_precision_score, f1_score
from tensorflow.keras.layers import Lambda


here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from dh_load_data import load_data


def run(p):

    (X_train, y_train), (X_valid, y_valid) = load_data()

    nc = 3
    size_emb = nc + 2
    size_n2v_emb = 128
    size_text_emb = 300
    dropout_rate = 0.1
    for i in range(1, 18):
        p[f"d{i}"] = dropout_rate

    inpt = tf.keras.layers.Input((np.shape(X_train)[1],))

    slice_n1 = inpt[:, :size_emb]
    slice_n2v_n1 = inpt[:, size_emb : size_emb + size_n2v_emb]
    slice_n2 = inpt[:, size_emb + size_n2v_emb : size_emb + size_n2v_emb + size_emb]
    slice_n2v_n2 = inpt[
        :,
        size_emb
        + size_n2v_emb
        + size_emb : size_emb
        + size_n2v_emb
        + size_emb
        + size_n2v_emb,
    ]
    slice_jc = inpt[
        :,
        size_emb
        + size_n2v_emb
        + size_emb
        + size_n2v_emb : size_emb
        + size_n2v_emb
        + size_emb
        + size_n2v_emb
        + 1,
    ]
    cursor = size_emb + size_n2v_emb + size_emb + size_n2v_emb + 1
    #     slice_sim_text = inpt[:, size_emb+size_n2v_emb+size_emb+size_n2v_emb+1:]
    slice_t_n1 = inpt[:, cursor : cursor + size_text_emb]
    slice_t_n2 = inpt[:, cursor + size_text_emb : cursor + size_text_emb * 2]

    # UMAP EMBEDDINGS
    l_dense1 = tf.keras.layers.Dense(p["u1"], activation=p["a1"])
    l_dropout1 = tf.keras.layers.Dropout(p["d2"])
    l_dense2 = tf.keras.layers.Dense(p["u2"], activation=p["a2"])
    l_dropout2 = tf.keras.layers.Dropout(p["d2"])
    l_dense3 = tf.keras.layers.Dense(p["u3"], activation=p["a3"])
    l_dropout3 = tf.keras.layers.Dropout(p["d3"])

    sub_n1_1 = l_dense1(slice_n1)
    sub_n1_do_1 = l_dropout1(sub_n1_1)
    sub_n1_2 = l_dense2(sub_n1_do_1)
    sub_n1_do_2 = l_dropout2(sub_n1_2)
    sub_n1_3 = l_dense3(sub_n1_do_2)
    sub_n1_do_3 = l_dropout3(sub_n1_3)

    sub_n2_1 = l_dense1(slice_n2)
    sub_n2_do_1 = l_dropout1(sub_n2_1)
    sub_n2_2 = l_dense2(sub_n2_do_1)
    sub_n2_do_2 = l_dropout2(sub_n2_2)
    sub_n2_3 = l_dense3(sub_n2_do_2)
    sub_n2_do_3 = l_dropout3(sub_n2_3)

    add_emb1 = tf.keras.layers.Concatenate()([sub_n1_do_3, sub_n2_do_3])
    add_emb1 = tf.keras.layers.Dense(p["u4"], activation=p["a4"])(add_emb1)
    add_emb1 = tf.keras.layers.Dropout(p["d4"])(add_emb1)

    # N2V EMBEDDINGS
    n2v_dense1 = tf.keras.layers.Dense(p["u5"], activation=p["a5"])
    n2v_dropout1 = tf.keras.layers.Dropout(p["d5"])
    n2v_dense2 = tf.keras.layers.Dense(p["u6"], activation=p["a6"])
    n2v_dropout2 = tf.keras.layers.Dropout(p["d6"])
    n2v_dense3 = tf.keras.layers.Dense(p["u7"], activation=p["a7"])
    n2v_dropout3 = tf.keras.layers.Dropout(p["d7"])

    sub_n1_1 = n2v_dense1(slice_n2v_n1)
    sub_n1_do_1 = n2v_dropout1(sub_n1_1)
    sub_n1_2 = n2v_dense2(sub_n1_do_1)
    sub_n1_do_2 = n2v_dropout2(sub_n1_2)
    sub_n1_3 = n2v_dense3(sub_n1_do_2)
    sub_n1_do_3 = n2v_dropout3(sub_n1_3)

    sub_n2_1 = n2v_dense1(slice_n2v_n2)
    sub_n2_do_1 = n2v_dropout1(sub_n2_1)
    sub_n2_2 = n2v_dense2(sub_n2_do_1)
    sub_n2_do_2 = n2v_dropout2(sub_n2_2)
    sub_n2_3 = n2v_dense3(sub_n2_do_2)
    sub_n2_do_3 = n2v_dropout3(sub_n2_3)

    add_n2v = tf.keras.layers.Concatenate()([sub_n1_do_3, sub_n2_do_3])
    add_n2v = tf.keras.layers.Dense(p["u8"], activation=p["a8"])(add_n2v)
    add_n2v = tf.keras.layers.Dropout(p["d8"])(add_n2v)

    # W2V EMBEDDINGS
    n2v_dense1 = tf.keras.layers.Dense(p["u9"], activation=p["a9"])
    n2v_dropout1 = tf.keras.layers.Dropout(p["d9"])
    n2v_dense2 = tf.keras.layers.Dense(p["u10"], activation=p["a10"])
    n2v_dropout2 = tf.keras.layers.Dropout(p["d10"])
    n2v_dense3 = tf.keras.layers.Dense(p["u11"], activation=p["a11"])
    n2v_dropout3 = tf.keras.layers.Dropout(p["d11"])

    sub_n1_1 = n2v_dense1(slice_t_n1)
    sub_n1_do_1 = n2v_dropout1(sub_n1_1)
    sub_n1_2 = n2v_dense2(sub_n1_do_1)
    sub_n1_do_2 = n2v_dropout2(sub_n1_2)
    sub_n1_3 = n2v_dense3(sub_n1_do_2)
    sub_n1_do_3 = n2v_dropout3(sub_n1_3)

    sub_n2_1 = n2v_dense1(slice_t_n2)
    sub_n2_do_1 = n2v_dropout1(sub_n2_1)
    sub_n2_2 = n2v_dense2(sub_n2_do_1)
    sub_n2_do_2 = n2v_dropout2(sub_n2_2)
    sub_n2_3 = n2v_dense3(sub_n2_do_2)
    sub_n2_do_3 = n2v_dropout3(sub_n2_3)

    add_w2v = tf.keras.layers.Add()([sub_n1_do_3, sub_n2_do_3])
    add_w2v = tf.keras.layers.Activation("sigmoid")(add_w2v)

    sub_jc_1 = tf.keras.layers.Dense(p["u12"], activation=p["a12"])(slice_jc)
    sub_jc_2 = tf.keras.layers.Dense(p["u13"], activation=p["a13"])(sub_jc_1)
    sub_jc_3 = tf.keras.layers.Dense(p["u14"], activation=p["a14"])(sub_jc_2)

    #     sub_st_1 = tf.keras.layers.Dense(8, activation='relu')(slice_sim_text)
    #     sub_st_2 = tf.keras.layers.Dense(8, activation='relu')(sub_st_1)
    #     sub_st_3 = tf.keras.layers.Dense(8, activation='relu')(sub_st_2)

    concat = tf.keras.layers.Concatenate()([add_emb1, add_n2v, add_w2v, sub_jc_3])

    dense1 = tf.keras.layers.Dense(p["u15"], activation=p["a15"])(concat)
    dense1 = tf.keras.layers.Dropout(p["d15"])(dense1)
    dense2 = tf.keras.layers.Dense(p["u16"], activation=p["a16"])(dense1)
    dense2 = tf.keras.layers.Dropout(p["d16"])(dense2)
    dense3 = tf.keras.layers.Dense(p["u17"], activation=p["a17"])(dense2)
    dense3 = tf.keras.layers.Dropout(p["d17"])(dense3)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(dense3)

    model = tf.keras.Model(inputs=[inpt], outputs=[out])

    # COMPILE
    lr = 0.001
    epochs = 10
    batch_size = 32

    opt = tf.keras.optimizers.Adam(lr=lr)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    thresh = 0.5
    y_pred_valid = model.predict(X_valid)
    predictions = (y_pred_valid >= thresh).astype(int).flatten()
    f1s = f1_score(predictions, y_valid.flatten())

    print("f1 score: ", f1s)

    return f1s


if __name__ == "__main__":
    from dh_problem import Problem

    sp = Problem.starting_point_asdict[0]
    run(sp)
