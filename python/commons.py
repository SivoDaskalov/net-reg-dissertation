from collections import namedtuple

Setup = namedtuple('Setup', ['label', 'true_coefficients', 'network', 'degrees',
                             'x_tune', 'y_tune', 'x_train', 'y_train', 'x_test', 'y_test'])
cross_validation_folds = 10