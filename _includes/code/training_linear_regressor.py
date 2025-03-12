from sacred import Experiment

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

ex = Experiment('training-linear-regressor')


@ex.config
def my_config():
  workers = 1
  test_size = 1 / 3
  split_random_state = 42


@ex.automain
def main(test_size, workers, split_random_state):
  dataset = load_boston()
  x_train, x_test, y_train, y_test = train_test_split(
      dataset.data, dataset.target,
      test_size=test_size,
      random_state=split_random_state)

  model = LinearRegression(n_jobs=workers)
  model.fit(x_train, y_train)

  print('train mse:', mse(y_train, model.predict(x_train)))
  print('test mse:', mse(y_test, model.predict(x_test)))

  print('y:', y_test[:5])
  print('p:', model.predict(x_test)[:5])
