from sacred import Experiment
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ex = Experiment('training-a-logistic-regression-model')

@ex.config
def my_config():
  workers = 1
  test_size = 1 / 3
  split_random_state = 42

@ex.automain
def main(test_size, workers, split_random_state):
  dataset = load_breast_cancer()
  x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target,
    test_size=test_size,
    random_state=split_random_state)

  model = LogisticRegression(n_jobs=workers)
  model.fit(x_train, y_train)

  print('train accuracy:', accuracy_score(y_train, model.predict(x_train)))
  print('test accuracy:', accuracy_score(y_test, model.predict(x_test)))

  print('y:', y_test)
  print('p:', model.predict(x_test))
