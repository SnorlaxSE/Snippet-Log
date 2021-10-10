from keras.models import load_model
import pdb


def rename_first_layer(model_path, new_model_path, flag='Flag'):

    model = load_model(model_path)
    model.layers[0].name = '{}_'.format(flag) + model.layers[0].name
    model.save(new_model_path)


if __name__ == "__main__":
    model_path = r'm1.acc.b16.h5'
    rename_first_layer(model_path, new_model_path='Flag_model.h5')
    pass