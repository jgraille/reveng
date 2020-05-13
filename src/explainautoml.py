import lime.lime_tabular
from src.h2owrapper import H2oWrapper
from matplotlib import pyplot as plt


class ExplainModel:

    def __init__(self, train_test_split, feature_names, class_names, categorical_features, categorical_names, row):
        self.training_data = train_test_split[0]
        self.data_row = train_test_split[1][row]
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.row = row

    def plot_explainer(self, exp):
        #plt.clf()
        exp.as_pyplot_figure()
        plt.title('Local explanation row {}'.format(self.row))
        plt.show()

    def explainer(self, model):
        try:
            h2o_wrapper = H2oWrapper(model=model, column_names=self.feature_names)
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.training_data,
                                                               mode='regression',
                                                               feature_names=self.feature_names,
                                                               categorical_features=self.categorical_features,
                                                               categorical_names=self.categorical_names,
                                                               discretize_continuous=True,
                                                               kernel_width=3)
            exp = explainer.explain_instance(self.data_row, h2o_wrapper.predict_proba, num_features=5)
            self.plot_explainer(exp)
            return exp
        except Exception as e:
            print('\nMethod ExplainModel.explainer did not work', e.__repr__())
