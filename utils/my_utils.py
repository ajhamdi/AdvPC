import os 
import imageio
import numpy as np
import pptk 
import glob
import pandas as pd

def gif_folder(data_dir, extension="jpg",duration=None):
    image_collection = []
    for img_name in sorted(glob.glob(data_dir + "/*." + extension)):
        image_collection.append(imageio.imread(img_name))
    if not duration:
        imageio.mimsave(os.path.join(data_dir, "animation.gif"), image_collection)
    else:
        imageio.mimsave(os.path.join(data_dir, "animation.gif"), image_collection, duration=duration)
# mesh = PyntCloud.from_file("/media/hamdiaj/D/mywork/sublime/vgd/3d/ModelNet40/airplane/test/airplane_0627.off")
def check_folder(data_dir):
    """
    checks if folder exists and create if doesnt exist 
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

def view_ptc(points,color,size,save_name=None,show_floor=False):
    ptc_nb = points.shape[0]
    viewer = pptk.viewer(points)
    colors = np.repeat(np.array([[color[0],color[1],color[2]]]),ptc_nb,axis=0)  ## RED = ADv  ...... BLUE = ORIGINAL
    viewer.attributes(colors)
    viewer.set(point_size=size,bg_color=(1,1,1,1),floor_color=(0,0,0,1),show_info=False,show_axis=False,show_grid=show_floor,r=5,phi=-np.radians(60),theta=np.radians(20))
    if save_name:
        viewer.capture(save_name)
    return viewer
def play_ptc(viewer,save_dir=None,duration=0.001):
    """
    play animation around the pptk.viewer object that is passed to it as viewer 
    if you want to save the vidoe : put your directory in "save_dir"
    duration determines how fast is teh resulting animation.gif 
    """
    viewer.play([(0,0,0,x,x/10.0,x) for x in range(1,9) ],repeat=True)
    if save_dir :
        viewer.record(save_dir, [(0, 0, 0, x, x/10.0, x) for x in range(1, 9)])
        viewer.close()
        gif_folder(save_dir, extension="png", duration=0.001)
def random_id(digits_nb=3,include_letters=True,only_capital=True,unique_digits=False):
    """
    generates random digits of length digits_nb and return the random string  . options include using letters or only numbers , unique or not uniqe digits, all capital letters or allow lowercase 
    
    Args:
        digits_nb : (int) the number of didits to be returned in the resulting 
        include_letters : (bool) flag wheathre to include letters or only numbers in the string 
        only_capital : (bool) flag wheathre to allow lowercase letters if include_letters==True
        unique_digits : (bool) flag wheathre to make all the digits unique or allow repetetion 
    Returns :
        string of random digits 
    """
    import numpy as np
    if include_letters:
        full_list = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if only_capital:
            full_list = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    else :
        full_list = list('0123456789')
    short_list = np.random.choice(full_list, digits_nb, replace= not unique_digits)
    mystr = ""
    for ii in short_list :
        mystr = mystr+ii
    return mystr 


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


class ListDict(object):
    """
    a class of list dictionary .. each element is a list , has the methods of both lists and dictionaries 
    idel for combining the results of some experimtns and setups 
    """
    def __init__(self, keylist_or_dict=None):
        # def initilize_list_dict(names):
        if isinstance(keylist_or_dict, list):
            self.listdict = {k: [] for k in keylist_or_dict}
        elif isinstance(keylist_or_dict, dict):
            if not isinstance(keylist_or_dict.values()[0],list):
                self.listdict = {k: [v] for k, v in keylist_or_dict.items()}
            else:
                 self.listdict = keylist_or_dict
        elif not keylist_or_dict:
            self.listdict = {}
        else:
            print("unkonwn type")

    def raw_dict(self):
        """
        returns the Dict object that is iassoicaited with the ListDict object 
        """
        return self.listdict

    def append(self, one_dict):
        for k, v in self.items():
            v.append(one_dict[k])
        return self

    def extend(self, newlistdict):
        for k, v in self.items():
            v.extend(newlistdict.raw_dict()[k])
        return self


    def partial_append(self, one_dict):
        for k, v in one_dict.items():
            self.listdict[k].append(v)
        return self


    def partial_extend(self, newlistdict):
        for k, v in newlistdict.items():
            self.listdict[k].extend(v)
        return self


    def combine(self, newlistdict):
        return ListDict(merge_two_dicts(self.raw_dict(), newlistdict.raw_dict()))
        # self.listdict = {**self.raw_dict(), **newlistdict.raw_dict()}

    def chek_error(self):
        for k, v in self.items():
            print(len(v), ":", k)
        return self


    def __getitem__(self, key):
        return self.listdict[key]
    def __str__(self):
        return str(self.listdict)
    def __len__(self):
        return len(self.listdict)
    def keys(self):
        return self.listdict.keys()
    def values(self):
        return self.listdict.values()
    def items(self):
        return self.listdict.items()


def log_setup(setup, setups_file):
    """
    update an exisiting CSV file or create new one if not exisiting using setup
    """
    setup_ld = ListDict(setup)
    if os.path.isfile(setups_file):
        old_ld = ListDict(pd.read_csv(setups_file, sep=",").to_dict("list"))
        old_ld.append(setup)
        setup_ld = old_ld
    pd.DataFrame(setup_ld.raw_dict()).to_csv(setups_file, sep=",", index=False)


def save_results(save_file, results):
    pd.DataFrame(results.raw_dict()).to_csv(save_file, sep=",", index=False)


def load_results(load_file):
    if os.path.isfile(load_file):
        df = pd.read_csv(load_file, sep=",")
        return ListDict(df.to_dict("list"))
    else:
        print(" ########## WARNING : no file names : {}".format(load_file))
        return None

# def parse_var(s):
#     """
#     Parse a key, value pair, separated by '='
#     That's the reverse of ShellArgs.

#     On the command line (argparse) a declaration will typically look like:
#         foo=hello
#     or
#         foo="hello world"
#     """
#     items = s.split('=')
#     key = items[0].strip()  # we remove blanks around keys, as is logical
#     if len(items) > 1:
#         # rejoin the rest:
#         value = '='.join(items[1:])
#     return (key, value)


# def parse_vars(items):
#     """
#     Parse a series of key-value pairs and return a dictionary
#     """
#     d = {}

#     if items:
#         for item in items:
#             key, value = parse_var(item)
#             d[key] = value
#     return d
