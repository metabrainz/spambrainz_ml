import numpy as np
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


def get_model() -> Model:
    main_input = Input(shape=(9,), name="main_input")
    email_input = Input(shape=(1,), dtype="int32", name="email_input")
    website_input = Input(shape=(1,), dtype="int32", name="website_input")
    bio_input = Input(shape=(512,), name="bio_input")

    email_embedding = Embedding(output_dim=256, input_dim=1024, input_length=1, name="email_embedding")(email_input)
    website_embedding = Embedding(output_dim=256, input_dim=1024, input_length=1, name="website_embedding")(website_input)
    bio_embedding = Embedding(output_dim=256, input_dim=1, input_length=512, name="bio_embedding")(bio_input)

    email_lstm = LSTM(32, name="email_lstm")(email_embedding)
    website_lstm = LSTM(32, name="website_lstm")(website_embedding)
    bio_lstm = LSTM(64, name="bio_lstm")(bio_embedding)

    merge = concatenate([website_lstm, email_lstm, bio_lstm, main_input], name="merge")

    dropout1 = Dropout(0.5, name="dropout_1")(merge)
    dense1 = Dense(64, activation="tanh", name="dense_1")(dropout1)
    dropout2 = Dropout(0.5, name="dropout_2")(dense1)
    dense2 = Dense(64, activation="tanh", name="dense_2")(dropout2)

    output = Dense(2, activation="softmax", name="output")(dense2)

    model = Model(inputs=[main_input, website_input, email_input, bio_input], outputs=[output])

    adam = Adam()

    model.compile(optimizer=adam, loss="mse", metrics=["acc"])

    return model


def train_model(model: Model, dataset: np.ndarray, callbacks: list = None) -> None:
    model.fit(
        {
            "main_input": dataset[:, 1:10],
            "email_input": dataset[:, 10],
            "website_input": dataset[:, 11],
            "bio_input": dataset[:, 12:]
        },
        to_categorical(dataset[:, 0], 2),
        epochs=3,
        batch_size=32,
        callbacks=callbacks,
        validation_split=0.1,
    )


def evaluate_model(model: Model, dataset: np.ndarray) -> None:
    print(model.evaluate(
        {
            "main_input": dataset[:, 1:10],
            "email_input": dataset[:, 10],
            "website_input": dataset[:, 11],
            "bio_input": dataset[:, 12:]
        },
        to_categorical(dataset[:, 0], 2),
        batch_size=32,
    ))
    print(model.metrics_names)


def load_model(path: str) -> Model:
    model = get_model()
    model.load_weights(path)
    return model


if __name__ == "__main__":
    import pickle
    import datetime
    from keras.callbacks import TensorBoard

    with open("../SENSITIVE/spambrainz_dataset.pickle", "rb") as f:
        training_data = pickle.load(f)

    tensorboard = TensorBoard(log_dir="./logs", write_graph=True, histogram_freq=0)

    m = get_model()
    train_model(m, training_data, [tensorboard])

    m.save_weights("snapshots/lodbrok-{}.h5py".format(datetime.datetime.now().isoformat()))

    # m = load_model("snapshots/lodbrok-2018-08-06T20:03:23.228350.h5py")

    with open("../SENSITIVE/spambrainz_dataset_eval.pickle", "rb") as f:
        eval_data = pickle.load(f)

    evaluate_model(m, eval_data)
