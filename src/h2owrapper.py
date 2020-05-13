import numpy
import pandas
import h2o


class H2oWrapper:
    # from https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
    def __init__(self, model, column_names):
        self.model = model
        self.column_names = column_names

    def predict_proba(self, this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = numpy.shape(this_array)
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)

        # We convert the numpy array that Lime sends to a pandas dataframe and
        # convert the pandas dataframe to an h2o frame
        self.pandas_df = pandas.DataFrame(data=this_array, columns=self.column_names)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)

        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame()
        # the first column is the class labels, the rest are probabilities for
        # each class
        #self.predictions = self.predictions.iloc[:, 1:].values
        self.predictions = self.predictions.iloc[:, 0:].values
        return self.predictions

