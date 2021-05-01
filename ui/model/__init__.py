import os
import joblib
from model.temp import Model


def get_violence_state():
    import ctypes
    hllDll = ctypes.WinDLL("User32.dll")
    return hllDll.GetKeyState(0x14)

def getp(res):
  return res[1] >= res[0]

def get_eval(m):
  model = Model(m)

  i = 0

  evaluation = get_violence_state()

  for video in os.listdir(os.path.join('tmp', 'current')):
    file = os.path.join('tmp', 'current', f'output_{str(i).zfill(4)}{os.path.splitext(video)[1]}')
    evaluation = evaluation or getp(model.predict([video]))

  return evaluation