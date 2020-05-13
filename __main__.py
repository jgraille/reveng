from cmd import Cmd
from pyfiglet import Figlet
from utils.utils import init_params
from __init__ import Database
from src.automl import AutoMl
from src.explainautoml import ExplainModel
from src.preprocessing import PreProcessing
from src.geneticlearn import GeneticProgramming
from src.preclustering import PreClustering
import pandas
import numpy

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


class Menu(Cmd):

    f = Figlet(font='slant')
    print(f.renderText(' Rev Eng\n'))
    print('----------------------------------------------')
    print('---------------------Menu---------------------')
    print('----------------------------------------------\n')

    print('* Computing preclustering / outliers enter--> cl\n')
    print('* Computing genetic programming enter--> gp\n')
    print('* Running an autml instance enter--> automl\n')
    print('* Press q to exit at anytime \n')
    prompt = 'reveng> '

    def __init__(self):
        config_db, config_automl, config_file1, config_file2, config_file3 = init_params()
        super(Menu, self).__init__()
        self.config_db = config_db
        self.config_automl = config_automl
        self.config_file = config_file1
        self.preprocessing = PreProcessing(config_file=self.config_file)

    def do_q(self, args=True):
        print('\n Closed RevEng')
        raise SystemExit

    def do_codb(self, args=True):
        print('\nConnecting to a database...')
        try:
            db = Database(config=self.config_db)
        except Exception as e:
            print('\nFailed to connect to the database:', e.__repr__())
            return 1
        print('Connection ok\n')

    def do_cl(self, args=True):
        try:
            outliers, data = self.preprocessing.outliers()
            print('\nOutliers done... ')
            if outliers:
                print('\nFound {} outliers'.format(len(outliers)))
            else:
                print('\nNo outliers found')
            scaled_data = self.preprocessing.frequency_and_scale()
            print('\nFrequency and scale done... ')
            clusters = PreClustering(data=scaled_data)
            labels_clusters = clusters.running()
            print('\nDbscan done... ')
            print('\nClusters labels: \n', numpy.unique(labels_clusters))
            clusters.display_clusters_outliers(outliers=outliers, data=data, labels_clusters=labels_clusters)
        except Exception as e:
            print('\nFailed to compute dbscan...', e.__repr__())

    def do_gp(self, args=True):
        try:
            train_test_split, feature_names = self.preprocessing.encoding_and_split()
            res = GeneticProgramming(x_train=train_test_split[0],
                                     y_train=train_test_split[2],
                                     excludes=self.preprocessing.excludes,
                                     header=self.preprocessing.header,
                                     config_file=self.config_file)
            pop, log, hof = res.calculate()
            print(pop[50])
        except Exception as e:
            print('\nFailed to compute gp...', e.__repr__())

    def do_automl(self, args=True):
        try:
            train_test_split, feature_names = self.preprocessing.encoding_and_split()
            reveng_model = AutoMl(train=train_test_split[0],
                                   test=train_test_split[1],
                                   labels_train=train_test_split[2],
                                   labels_test=train_test_split[3],
                                   feature_names=feature_names,
                                   categforical_features=self.preprocessing.categorical_features)
            reveng_model.running(config_automl=self.config_automl)
            reveng_model.h2oshellinformations(model=reveng_model.model, test_h2o_df=reveng_model.test_h2o_df, automl=True)
            exp = ExplainModel(train_test_split=train_test_split,
                               feature_names=feature_names,
                               class_names=self.preprocessing.class_names,
                               categorical_features=self.preprocessing.categorical_features,
                               categorical_names=self.preprocessing.categorical_names,
                               row=3).explainer(model=reveng_model.model)
            print(exp.as_list())
            reveng_model.close()
        except Exception as e:
            print('\nFailed to connect to h2o / process the file...', e.__repr__())


def main():
    menu = Menu().cmdloop()


if __name__ == '__main__':
    main()
