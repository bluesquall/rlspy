from distutils.core import setup

package_list = ['rlspy', ]
classifier_list = ['License :: OSI Approved :: MIT License',]

setup(name = 'rlspy', version = '0.0.1',
      description = 'Tools and examples for RLS estimation.',
      author = 'M J Stanway', author_email = 'm.j.stanway@alum.mit.edu',
      packages = package_list, classifiers = classifier_list,
    )
