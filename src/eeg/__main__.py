from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from arl_eegmodels.EEGModels import EEGNet


def main():
    model: Model = EEGNet(4)
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    
    checkpoint_filepath = 'tmp/checkpoint'
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_filepath, verbose=1, save_best_only=True
    )

    # history = model.fit()

    # Load optimal weights
    model.load_weights(checkpoint_filepath)



if __name__ == "__main__":
    main()
