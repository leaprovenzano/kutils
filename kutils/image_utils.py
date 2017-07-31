from PIL import Image
import os
import sys


class ImageBatchGenerator(object):
    """given a tied image and labelgenerator 
    generates from both, good for multi labeled 
    images that are not split into class directories
    
    Attributes:
        batch_size (int)
        imgen : Image generator with a next  & reset method must be tied to labelgen
        labelgen : Label generator with a next & reset method must be tied to labelgen
        n (int): total num of samples
    """

    def __init__(self, imgen, labelgen):
        self.imgen = imgen
        self.labelgen = labelgen
        self.n = self.imgen.n
        self.batch_size = self.imgen.batch_size

    def reset():
        self.imgen.reset()
        self.labelgen.reset()

    def __next__(self):
        return self.next()

    def next(self):
        while True:
            return self.imgen.next(), self.labelgen.next()[1]


class ImageScaler(object):
    def __init__(self, use_std=True, channel_axis=3, verbose=True):
        self.n_seen = 0
        self.use_std = use_std
        self.axis = tuple([i for i in range(4) if i != channel_axis])
        self.verbose = verbose
        self._std = None
        self._mean = None

    def print_stats(self, ds, ds_name):
        """Print some basic stats about a dataset (as numpy array)
        """
        print("{} - min: {:0.3f} max: {:0.3f} mean: {:0.3f} std: {:0.3f}".format(ds_name,
                                                                                 ds.min(), ds.max(), ds.mean(), ds.std()))

    def fit(self, X):
        self.n_seen = X.shape[0]
        if self.verbose:
            print('fitting {} examples..'.format(self.n_seen))
        self._mean = X.mean(axis=self.axis)
        if self.use_std:
            self._std = X.std(axis=self.axis)
        if self.verbose:
            print('mean: {}'.format(self._mean))
            if self.use_std:
                print('std: {}'.format(self._std))


    def transform(self, X, X_name='dataset'):
        scaled = (X - self._mean) / self._std
        if self.verbose:
            self.print_stats(scaled, X_name)
        return scaled

    def inverse_transform(self, X, X_name='dataset'):
        de_scaled = (X * self._std) + self._mean
        if self.verbose:
            self.print_stats(de_scaled, X_name)
        return de_scaled



# fix exif if it exists...
def fix_missing_exif(directories, v=True, prog_bar=None):
    """remove piexifs from images
    takes a list of keras image generators
    note that these are directory iterators
    so you must call flow_from_directory method 
    on the generators first.
    
    Args:
        image_generators (TYPE): Description
        v (bool, optional): verbose opt
        prog_bar (None, optional): if provided uses a progress bar such as tqdm

    """
    try:
        import piexif
    except ImportError:
        print('You must install the piexif library "pip install piexif"')
    for d in directories:
        filenames =os.listdir(d)
        if v:
            print('removing exifs from {} images at {}...'.format(len(filenames),
                                                                  d))
        if prog_bar:
            filenames =  prog_bar(filenames)
        for imfile in filenames:
            p = os.path.join(d, imfile)
            try:
                piexif.remove(p)
            except Exception:
                print('caught error InvalidImageDataError for file at {} ignoring...')
                pass
    if v:
        print('complete!')


def crop_shortest(im):
    """ crop an image to it's shortest side using the shortest side """
    width, height = im.size   # Get dimensions
    max_side = max([width, height])
    min_side = min([width, height])
    c = (max_side - min_side) / 2
    left = (0 + (c * (width is max_side)))
    top = (0 + (c * (height is max_side)))
    right = (width - (c * (width is max_side)))
    bottom = (height - (c * (height is max_side)))
    return im.crop((left, top, right, bottom))


def resize_dir(orig_root, im_size=(256, 256), suffix='_resized', crop=None, prog_bar=None):
    """given some directory of images, make a new directory containing resized
    versions of those images. 
    
    Args:
        orig_root (str): directory of images
        im_size (tuple, optional): size for resized imgs, defaults to (256, 256)
        suffix (str, optional): name resized directories with this suffix
        crop (bool, optional): if true will crop to square using the shortest side before resize
        prog_bar (None, optional): if provided uses a progress bar such as tqdm
    """
    rs_root = orig_root + suffix
    def makefile_or_whatever(f):
        try:
            f()
        except FileExistsError:
            pass

    makefile_or_whatever(lambda: os.mkdir(rs_root))
    orig_to_rs = lambda p: p.replace(orig_root, rs_root)
    for d in os.listdir(orig_root):
        if not d.startswith('.'):
            p, _, im_files = next(os.walk(os.path.join(orig_root, d)))
            print('resizing {} images from {} to {}'.format(
                len(im_files), p, orig_to_rs(p)))
            
            f_iter = enumerate(im_files)
            if prog_bar:
                f_iter = prog_bar(f_iter)
            for i, imfile in f_iter:
                f = os.path.join(p, imfile)
                if not os.path.exists(orig_to_rs(f)):
                    try:
                        im = Image.open(f)
                        if crop:
                            imx = crop(im)
                        else:
                            imx = im
                        imx = imx.resize(im_size, resample=Image.LANCZOS)
                        try:
                            makefile_or_whatever(
                                lambda: imx.save(orig_to_rs(f)))
                        except OSError:
                            makefile_or_whatever(lambda: os.mkdir(
                                orig_to_rs(f)[:orig_to_rs(f).rfind('/')]))
                            makefile_or_whatever(
                                lambda: imx.save(orig_to_rs(f)))
                    except OSError:
                        pass

