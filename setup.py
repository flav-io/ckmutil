from setuptools import setup, find_packages


setup(name='ckmutil',
      version='0.1',
      author='David M. Straub',
      author_email='david.straub@tum.de',
      url='https://davidmstraub.github.io/ckmutil',
      description='Useful functions for dealing with quark and lepton mixing matrices.',
      long_description="""``ckmutil`` is a package containing useful functions
      to deal with the Cabibbo-Kobayashi-Maskawa (CKM) quark mixing matrix or
      the Pontecorvo-Maki-Nakagawa-Sakata (PMNS) lepton mixing matrix
      in high energy physics.""",
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'setuptools>=3.3'],
    )
