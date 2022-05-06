
import pandas as pd
import numpy as np
NUM_FOLDS_OUTTER = 2

def data_loader():

    d0 = 'datasets/elusage.csv'
    d1 = 'datasets/fri_c0_250_5.csv'
    d2 = 'datasets/hill-valley.csv'
    d3 = 'datasets/image-segmentation.csv'
    d4 = 'datasets/iris.csv'
    d5 = 'datasets/libras.csv'
    d6 = 'datasets/molec-biol-promoter.csv'
    d7 = 'datasets/monks-1.csv'
    d8 = 'datasets/no2.csv'
    d9 = 'datasets/plant-shape.csv'
    d10 = 'datasets/synthetic-control.csv'
    d12 = 'datasets/chess-krvkp.csv'
    d13 = 'datasets/diggle_table_a2.csv'
    d14 = 'datasets/disclosure_z.csv'
    d15 = 'datasets/lowbwt.csv'
    d16 = 'datasets/mammographic.csv'
    d17 = 'datasets/pm10.csv'

    d18 = 'datasets/car.csv'
    d19 = 'datasets/cloud.csv'
    d20 = 'datasets/chatfield_4.csv'
    d21 = 'datasets/schizo.csv'

    d22 = 'datasets/post-operative.csv'
    d23 = 'datasets/flags.csv'
    d24 = 'datasets/autos.csv'
    d25 = 'datasets/credit-approval.csv'


    data_names = [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10, d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25]
    data_frames = []
    for csv_name in data_names:
        temp_df = pd.read_csv(csv_name)
        temp_df = temp_df.set_axis([*temp_df.columns[:-1], 'class'], axis=1, inplace=False)

        temp_df = temp_df.fillna(temp_df.mean())
        for col_name in temp_df.columns:
            if temp_df[col_name].dtype == "object":
                temp_df[col_name] = pd.Categorical(temp_df[col_name])
                temp_df[col_name] = temp_df[col_name].cat.codes
        X = temp_df.drop('class', axis=1)
        y = temp_df['class']
        data_frames.append((X, y, len(pd.unique(temp_df['class']))))

    return data_frames


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def one_hot(y_test, n_class):
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1, 1)
    y_test = indices_to_one_hot(y_test, n_class)

    return y_test


class Data:
    pass

