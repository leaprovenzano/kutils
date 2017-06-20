from PIL import Image
import os, sys



class ImageScaler(object):
    def __init__(self, use_std=True, channel_axis=3, verbose=True):
        self.n_seen=0
        self.use_std = use_std
        self.axis = tuple([i for i in range(4) if i !=channel_axis])
        self.verbose=verbose
        self._std = None
        self._mean = None
        
    def print_stats(self, ds, ds_name):
        print("{} - min: {:0.3f} max: {:0.3f} mean: {:0.3f} std: {:0.3f}".format(ds_name, ds.min(), ds.max(), ds.mean(), ds.std()))
        
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
        scaled = (X - self._mean)/self._std
        if self.verbose:
            self.print_stats(scaled, X_name)
        return scaled
    
    def inverse_transform(self, X, X_name='dataset'):
        de_scaled = (X * self._std) + self._mean
        if self.verbose:
            self.print_stats(de_scaled, X_name)
        return de_scaled        
        



# fix exif if it exists...
def fix_missing_exif(image_generators, v=True):
    """remove piexifs from images
    takes a list of keras image generators
    note that these are directory iterators
    so you must call flow_from_directory method 
    on the generators first.
    """
    import piexif
    for gen in image_generators:
        if v: 
            print('removing exifs from {} images at {}...'.format(gen.samples, 
                                                                    gen.directory))
        for imfile in gen.filenames:
            p = os.path.join(gen.directory, imfile)
            try: 
                piexif.remove(p)
            except Exception:
                print('caught error InvalidImageDataError for file at {} ignoring...')
                pass
    if v: print('complete!')


def crop_shortest(im):
    width, height = im.size   # Get dimensions
    max_side = max([width, height])
    min_side = min([width, height])
    c = (max_side - min_side)/2
    left = (0 + (c * (width is max_side)))
    top = (0 + (c * (height is max_side)))
    right = ( width - (c * (width is max_side)))
    bottom = (height - (c * (height is max_side)))
    return im.crop((left, top, right, bottom))



def resize_crop_dir(orig_root, im_size=(256, 256), suffix='_rs', resample=Image.LANCZOS):
    rs_root = orig_root + suffix

    def makefile_or_whatever(f):
        try:
            f()
        except FileExistsError:
            pass
        
    makefile_or_whatever(lambda : os.mkdir(rs_root))
    orig_to_rs = lambda p : p.replace(orig_root, rs_root)
    for d in os.listdir(orig_root):
        if not d.startswith('.'):
            p,_, im_files = next(os.walk(os.path.join(orig_root, d))) 
            print('resizing {} images from {} to {}'.format(len(im_files), p, orig_to_rs(p)))
            for i, imfile in enumerate(im_files):
                if imfile.endswith('.jpg'):
                    f = os.path.join(p, imfile)
                    if not os.path.exists(orig_to_rs(f)):
                        try:
                            im = Image.open(f)
                            imx = crop_shortest(im)
                            imx = imx.resize(im_size, resample=resample)
                            try:
                                makefile_or_whatever(lambda : imx.save(orig_to_rs(f), 'jpeg'))
                            except OSError:
                                makefile_or_whatever(lambda : os.mkdir(orig_to_rs(f)[:orig_to_rs(f).rfind('/')]))
                                makefile_or_whatever(lambda : imx.save(orig_to_rs(f), 'jpeg'))
                        except OSError:
                            pass





        
        
        