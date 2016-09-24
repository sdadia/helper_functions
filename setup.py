from distutils.core import setup

setup(
        name            =   'helper_functions',
        # packages        =   ['helper_functions'], # this must be the same as the name above
        version         =   '2.0.0',
        py_modules      =   ['helper_functions'],
        author          =   'Sahil Dadia',
        author_email    =   'dadiasahil94@yahoo.in',
        url             =   'https://github.com/sdadia/helper_functions.git',# use the URL to the github repo
        description     =   'A simple module of simple function, for opencv and python3',
        license         =   'LICENSE.txt',
        keywords        =   ['opencv', 'helper', 'scripts'], # arbitrary keywords
        classifiers     =   [
                            'Topic :: Utilities',
                            'Development Status :: 3 - Alpha',
                            'Intended Audience :: Developers',
                            'License :: OSI Approved :: MIT License',
                            'Operating System :: OS Independent',
                            'Programming Language :: Python :: 2',
                            'Natural Language :: English'
                            ]
    )
