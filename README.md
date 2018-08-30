# AtrialFibrePlugin
Atrial Fibre Generation

### Setup Notes

1. Use Miniconda3 with Eidolon, it's just the easy way to setup.

2. Eidolon is currently compiled for Python 3.6 so make sure that's installed instead of the current Python 3.7.

3. Numpy with MKL has problems with concurrency which will cause random hangs during processing. The solution is to
setup Miniconda without MKL in a separate environment:

       conda create -n nomkl python=3.6.3 nomkl numpy scipy cython imageio pyqt six pandas
    
4. The plugin directory must be placed in `$HOME/.eidolon/plugins` or some other directory given on the command line:

       $EIDOLONHOME/run.sh --setting userplugindir /dir/containing/plugindir ...
    
