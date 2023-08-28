import subprocess

try:
    subprocess.run(['python', 'feature_engineering.py'])
except subprocess.CalledProcessError:
    print("Error al ejecutar 'feature_engineering.py'")

try:
    subprocess.run(['python', 'train.py'])
except subprocess.CalledProcessError:
    print("Error al ejecutar 'train.py'")
