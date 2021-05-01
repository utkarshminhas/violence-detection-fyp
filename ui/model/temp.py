import joblib
import time

class Model:
  def __init__(self, m):
    time.sleep(2)
    print('Class created')

  def predict(self, x):
    return [1, 0]

def gen():
  model = Model()
  joblib.dump(model, 'model.h5')

if __name__ == '__main__':
  gen()