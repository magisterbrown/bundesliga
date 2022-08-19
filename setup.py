from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    deps = f.readlines()
    global_pack = list(filter(lambda x:not ('@ file' in x), deps))

setup(name='bundespack',
      version='1.0',
      description='',
      author='magisterbrownie',
      author_email='magisterbrownie@gmail.com',
      url='',
      packages=find_packages(),
      install_requires=global_pack
     )
