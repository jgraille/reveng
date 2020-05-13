import h2o
from h2o.automl import H2OAutoML
import gc
import time
import matplotlib.pyplot as plt


class AutoMl:

    def __init__(self, train, test, labels_train, labels_test, feature_names, categforical_features):
        """
        h2o checks wether an instance exists on the server or not / cant specify startH20=True in python anyway.
        http: // docs.h2o.ai / h2o / latest - stable / h2o - docs / starting - h2o.html
        """
        h2o.init()
        self.train = train
        self.test = test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.feature_names = feature_names
        self.categorical_features = categforical_features
        self._model = None
        self._test_h2o_df = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if isinstance(value, H2OAutoML):
            self._model = value
        else:
            raise TypeError('Value is not a model h2oautoml')

    @property
    def test_h2o_df(self):
        return self._test_h2o_df

    @test_h2o_df.setter
    def test_h2o_df(self, value):
        self._test_h2o_df = value

    def h2oframe(self):
        """
        from https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
        """
        train_h2o_df = h2o.H2OFrame(self.train)
        train_h2o_df.set_names(self.feature_names)
        train_h2o_df['class'] = h2o.H2OFrame(self.labels_train)
        #train_h2o_df['class'] = train_h2o_df['class'].asfactor()

        test_h2o_df = h2o.H2OFrame(self.test)
        test_h2o_df.set_names(self.feature_names)
        test_h2o_df['class'] = h2o.H2OFrame(self.labels_test)
        #test_h2o_df['class'] = test_h2o_df['class'].asfactor()

        for feature in self.categorical_features:
            train_h2o_df[feature] = train_h2o_df[feature].asfactor()
            test_h2o_df[feature] = test_h2o_df[feature].asfactor()
        return train_h2o_df, test_h2o_df

    def running(self, config_automl):
        try:
            train_h2o_df, self.test_h2o_df = self.h2oframe()
            """
            gc.collect()
            
            from h2o.estimators.random_forest import H2ORandomForestEstimator
            from h2o.estimators.gbm import H2OGradientBoostingEstimator
            model = H2ORandomForestEstimator(ntrees=1,
                                             nfolds=5,
                                             fold_assignment="Modulo",
                                             keep_cross_validation_predictions=True,
                                             seed=1)
            
            model = H2OGradientBoostingEstimator(nfolds=3,
                                                 distribution='gaussian',
                                                 fold_assignment='Random')
            """
            self._model = H2OAutoML(**config_automl)
            self._model.train(x=self.feature_names, y='class', training_frame=train_h2o_df)
        except Exception as e:
            print('\nProcedure AutoMl.running did not work', e.__repr__())

    def h2oshellinformations(self, model, test_h2o_df, automl):
        try:
            if automl:
                print('\nModel leaderboard: ', model.leaderboard)
                model = model.leader
            print('\nThe model used is: ', model.model_id)
            print('\nTime in min: ', model.run_time * 0.0001 * (1 / 60))
            print(model.varimp(use_pandas=True))
            print('\nPerform model_performance method on test data: ')
            time.sleep(5)
            print(model.model_performance(test_data=test_h2o_df))
            time.sleep(5)
        except Exception as e:
            print('\nMethod AutoMl.h2oshellinformations did not work', e.__repr__())

    @staticmethod
    def close():
        h2o.shutdown(prompt=True)
