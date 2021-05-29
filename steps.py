import numpy as np
import os
from raise_utils.learners import FeedforwardDL, Autoencoder
from raise_utils.data import DataLoader, Data
from raise_utils.hyperparams import DODGE
from raise_utils.interpret import DODGEInterpreter
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transform import Transform
from raise_utils.hooks import Hook
from imblearn.over_sampling import RandomOverSampler


# Dataset filenames
file_dic = {'ivy':     ['ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv'],
            'lucene':  ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
            'poi':     ['poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
            'synapse': ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],
            'velocity': ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
            'camel': ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
            'jedit': ['jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv'],
            'log4j': ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
            'xalan': ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
            'xerces': ['xerces-1.2.csv', 'xerces-1.3.csv', 'xerces-1.4.csv']
            }

# For the Wang et al. experiments
file_dic_wang = {"ivy":     ["ivy-1.4.csv", "ivy-2.0.csv"],
                 "lucene":  ["lucene-2.0.csv", "lucene-2.2.csv"],
                 "lucene2": ["lucene-2.2.csv", "lucene-2.4.csv"],
                 "poi":     ["poi-1.5.csv", "poi-2.5.csv"],
                 "poi2": ["poi-2.5.csv", "poi-3.0.csv"],
                 "synapse": ["synapse-1.0.csv", "synapse-1.1.csv"],
                 "synapse2": ["synapse-1.1.csv", "synapse-1.2.csv"],
                 "camel": ["camel-1.2.csv", "camel-1.4.csv"],
                 "camel2": ["camel-1.4.csv", "camel-1.6.csv"],
                 "xerces": ["xerces-1.2.csv", "xerces-1.3.csv"],
                 "jedit": ["jedit-3.2.csv", "jedit-4.0.csv"],
                 "jedit2": ["jedit-4.0.csv", "jedit-4.1.csv"],
                 "log4j": ["log4j-1.0.csv", "log4j-1.1.csv"],
                 "xalan": ["xalan-2.4.csv", "xalan-2.5.csv"]
                 }

# Configurations for each stage of GHOST-v2.
process_configs = {
    'acde': {
        'weighted_loss': False,
        'wfo': True,
        'dodge': True,
        'smote': True,
        'ultrasample': False
    },
    'abce': {
        'weighted_loss': True,
        'smote': True,
        'wfo': False,
        'dodge': True,
        'ultrasample': False
    },
    'abcd': {
        'weighted_loss': True,
        'smote': True,
        'dodge': False,
        'wfo': True,
        'ultrasample': False
    },
    'vanilla': {
        'weighted_loss': False,
        'wfo': False,
        'dodge': False,
        'smote': False,
        'ultrasample': False
    },
    'weighted_losses': {
        'weighted_loss': True,
        'wfo': False,
        'dodge': False,
        'smote': False,
        'ultrasample': False
    },
    'wfo_only': {
        'weighted_loss': True,
        'wfo': True,
        'dodge': False,
        'smote': False,
        'ultrasample': False
    },
    'tuning': {
        'weighted_loss': False,
        'wfo': False,
        'dodge': True,
        'smote': False,
        'ultrasample': False
    },
    'wfo_with_tuning': {
        'weighted_loss': True,
        'wfo': True,
        'dodge': True,
        'smote': False,
        'ultrasample': False
    },
    'ghost': {
        'weighted_loss': True,
        'wfo': True,
        'dodge': True,
        'smote': True,
        'ultrasample': False
    },
    'ghost-v2': {
        'weighted_loss': True,
        'wfo': True,
        'dodge': True,
        'smote': True,
        'ultrasample': True
    }
}


def run(data: Data, name: str, config: dict):
    '''
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    '''
    if config.get('ultrasample', False):
        # Apply WFO
        transform = Transform('wfo')
        transform.apply(data)

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(*data)
            ae.fit()

            loss = ae.model.history.history['loss'][-1]

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    if config.get('dodge', False):
        # Tune the hyper-params
        dodge_config = {
            'n_runs': 10,
            'data': [data],
            'metrics': ['f1', 'd2h', 'pd', 'pf', 'prec'],
            'learners': [],
            'log_path': './ghost-log/',
            'transforms': ['standardize', 'normalize', 'minmax'] * 30,
            'random': True,
            'name': name
        }

        for _ in range(30):
            wfo = config.get('wfo', True)
            smote = config.get('smote', True)
            weighted = config.get('weighted_loss', True)

            dodge_config['learners'].append(
                FeedforwardDL(weighted=weighted, wfo=wfo, smote=smote,
                              random={'n_units': (
                                  2, 6), 'n_layers': (2, 5)},
                              n_epochs=50)
            )

        dodge = DODGE(dodge_config)
        dodge.optimize()
        return

    # Otherwise, it's one of the untuned approaches.
    elif config.get('wfo', False):
        learner = FeedforwardDL(weighted=True, wfo=True,
                                smote=True, n_epochs=50)
        learner.set_data(*data)
        learner.fit()

    elif config.get('weighted_loss', False):
        learner = FeedforwardDL(weighted=True, wfo=False,
                                smote=False, n_epochs=50)
        learner.set_data(*data)
        learner.fit()

    else:
        learner = FeedforwardDL(
            weighted=False, wfo=False, smote=False, n_epochs=50, random={'n_layers': (1, 5), 'n_units': (5, 20)})
        learner.set_data(*data)
        learner.fit()

    # Get the results.
    preds = learner.predict(data.x_test)
    m = ClassificationMetrics(data.y_test, preds)
    m.add_metrics(['f1', 'd2h', 'pd', 'pf', 'prec'])
    results = m.get_metrics()
    return results


def run_experiment(dic, filename: str, config_name: str):
    '''
    Runs 20 runs of a specific config on a certain file.

    :param {str} filename - The filename. Must not include a path.
    :param {dict} config_name - The name of the config. Must be in `process_configs`.
    '''
    config = process_configs[config_name]
    name = filename

    if config['dodge']:
        name += f'-{config_name}'

    base_path = '../DODGE Data/defect/'

    # Get the dataset, and binarize it.

    def _binarize(x, y): y[y > 1] = 1
    dataset = DataLoader.from_files(
        base_path=base_path, files=dic[filename], hooks=[Hook('binarize', _binarize)])

    f = open('runs.txt', 'a')
    print('[Imbalance]:', filename, '-', round(
        sum(dataset.y_train) / len(dataset.y_train), 3) * 100,
        '|', sum(dataset.y_test) / len(dataset.y_test), file=f)
    f.close()

    # If we're using DODGE, then it prints a file that we interpret and runs 10 times.
    if config['dodge']:
        run(dataset, name, config)

        # Now we need to interpret it.
        interp = DODGEInterpreter(files=[f'./ghost-log/{name}.txt'], max_by=0, metrics=[
                                  'f1', 'd2h', 'pd', 'pf', 'prec'])
        result = interp.interpret()

        return result[name + '.txt']['f1']

    else:
        # We need to run the loop ourselves.
        # Hold the results.
        results = []
        for i in range(10):
            np.random.seed(np.random.randint(2, 2e4))
            dataset = DataLoader.from_files(
                base_path=base_path, files=dic[filename], hooks=[Hook('binarize', _binarize)])
            result = run(dataset, name, config)
            results.append(result[0])

        return results


def run_all_experiments():
    """
    Runs all experiments 10 times each.
    """
    # DODGE needs a directory called `./ghost-log/`, or `./ghost-log-wang/` depending
    # on the datasets used.
    if 'ghost-log' not in os.listdir('.'):
        os.mkdir('./ghost-log')

    for file in file_dic:
        f = open('runs.txt', 'a')
        print(f'{file}:', file=f)
        print('=' * len(f'{file}:'), file=f)

        for config_name in process_configs:
            if config_name not in ['acde', 'abce', 'abcd']:
                continue
            print(f'{config_name}: ', end='', file=f)

            result = run_experiment(file_dic, file, config_name)
            print(result, file=f)
            f.flush()

        print('', file=f)

    for file in file_dic_wang:
        f = open('runs.txt', 'a')
        print(f'{file}:', file=f)
        print('=' * len(f'{file}:'), file=f)

        for config_name in process_configs:
            if config_name not in ['acde', 'abce', 'abcd']:
                continue
            print(f'{config_name}: ', end='', file=f)

            result = run_experiment(file_dic_wang, file, config_name)
            print(result, file=f)
            f.flush()

        print('', file=f)

    print('Done.', file=f)
    print('Done.')

    f.close()


if __name__ == '__main__':
    run_all_experiments()
