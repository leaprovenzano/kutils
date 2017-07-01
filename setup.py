from setuptools import setup, find_packages

setup(name='kutils',
      version='0.0.1',
      description='Helper utils for the Keras library as well as basic data science helpers.',
      url='https://github.com/leaprovenzano/kutils',
      author='Lea Provenzano',
      author_email='leaprovenzano@gmail.com',
      license='MIT',
      classifiers=[

          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'License :: OSI Approved :: MIT License',

      ],
      install_requires=['keras',
                        'numpy',
                        'pillow',
                        'matplotlib',
                        'piexif',
                        'scikit-learn'
                        ],
      )
