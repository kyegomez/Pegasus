from setuptools import setup, find_packages
# 

setup(
  name = 'pegasusX',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'pegasus - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/pegasus',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers',
    "Prompt Engineering"
  ],
    install_requires=[
        'oceandb',
        'numba',
        'joblib',
    ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)