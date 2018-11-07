'''
Module data_preprocess.py contains methods to:
    extract faces and landmarks from images,
    handlabel extracted faces,
    distribure images in classname folders.
'''

# Necessary imports
from utils import *
import dlib
from ipywidgets import IntProgress, interact
from IPython.display import display
import ipywidgets as widgets


# Dictionary with binary class labels (for 2 problems).
classes_binary = {
    'smile': ['calm', 'smile'],
    'mouth': ['closed', 'open']
}

# Dictionary with categorical classes.
classes_categorical = {
    0: 'calm_closed',
    1: 'calm_open',
    2: 'smile_closed',
    3: 'smile_open'
}


def shape_to_array(shape, n_points=68, dtype='int'):
    '''
    Get numpy array of landmark points from dlib shape object.
    
    Parameters
    ----------
    shape : dlib shape object
    n_points : int
        Number of points detected.
    dtype : type
        Type for the output array values.
    
    Returns
    -------
    points : ndarray of shape (n_points, 2)
    '''
    
    # Initialize output array
    points = np.zeros((n_points, 2), dtype)
    # Copy values
    for i in range(n_points):
        points[i, 0] = shape.part(i).x
        points[i, 1] = shape.part(i).y
    return points


def crop_by_landmarks(img, points, pad=5):
    '''
    Crop a box with face and all facial landmarks from input image.
    
    Parameters
    ----------
    img : ndarray
        Input image as numpy array.
    points : ndarray of shape (?, 2)
        Array with coordinates of facial landmarks.
    pad : int  
    
    Returns
    -------
    face, points_shifted: tuple
        Tuple of cropped image and shifted landmarks coordinates.
    '''
    
    # Get boundaries
    left, top = points.min(axis=0)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right, bottom = points.max(axis=0)
    right = min(img.shape[1], right + pad)
    bottom = min(img.shape[0], bottom + pad)
    # Crop image
    face = img[top:bottom, left:right, :]
    points_shifted = points - [left, top]
    return face, points_shifted


def detect_faces(img, predictor_path, plot=True, plot_landmarks=True):
    '''
    Detects faces on image and landmarks for each of the face.
    Otionally, plot the image with all detected faces and landmarks.
    
    Parameters
    ----------
    img : PIL image
    predictor_path : str
        Path to a trained dlib face detector.
    plot : bool
    plot_landmarks : bool
    
    Returns
    -------
    faces : list
        List of extracted face images.
    landmarks : list
        List of shifted landmarks coordinates arrays.
    '''
    
    # Detect face(s) and landmarks on image
    face_detector = dlib.get_frontal_face_detector()
    # Read trained landmarks predictor model
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces
    dets = face_detector(img, 1)

    if plot:
        # Plot image
        fig, ax = plt.subplots(figsize=(15,10))
        
        ax.imshow(img)
        ax.axis('off')

    # Iterate over all detected faces
    faces = []
    landmarks = []
    for i, d in enumerate(dets):
        # Get face boundaries
        left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
        
        # Predict landmarks
        shape = predictor(img, d)
        points = shape_to_array(shape)
        
        if plot:
            # Plot boundaries
            ax.add_patch(plt.Rectangle((left, bottom), right-left, top-bottom, fill=False, color='green'))
            ax.annotate(str(i), (left+5, bottom-5), color='lime')
            if plot_landmarks:
                plt.scatter(points[:, 0], points[:, 1], c='lime', s=1)
            
        # Crop face
        face, _ = crop_by_landmarks(img, points)
        faces.append(face)
        landmarks.append(points)

    if plot:
        plt.title('Faces detected: %d' % len(dets))
        plt.show()
    
    return faces, landmarks


def process_folder(path_in, path_out, path_to_annotations, predictor_path):
    '''
    Detects faces and landmarks on every image in the folder,
    and saves cropped faces to the output folder. Landmarks are
    saves to a DataFrame, or appended to an existing one.
    
    Parameters
    ----------
    path_in : str
        Folder with input images.
    path_out : str
        Ouput folder for extracted faces.
    path_to_annotations : str
        Path to annotations csv file.
    predictor_path : str
        Path to a trained dlib face detector.
    Returns
    -------
    '''
    # Create output dir, if needed.
    if not os.path.exists(os.path.abspath(path_out)):
        os.makedirs(os.path.abspath(path_out))
    
    # Get list of image pathes.
    imgs = [os.path.join(path_in, filename) for filename in os.listdir(path_in)]

    # Configure dlib face detector and landmarks predictor
    face_detector = dlib.get_frontal_face_detector()
    # Read trained landmarks predictor model
    predictor = dlib.shape_predictor(predictor_path)

    landmarks = {}
    count = 0

    # Progress bar
    bar = IntProgress(value=1, min=1, max=len(imgs), step=1)
    print('Processing folder: ' + path_in)
    display(bar)
    
    # Read the annotations file
    if os.path.exists(path_to_annotations):
        annotations = True
        df = pd.read_csv(path_to_annotations, header=0, index_col=0)
    else:
        annotations = False

    for filename in imgs:
        # Update progress bar
        bar.description = '%d / %d' % (bar.value, len(imgs))

        # Try to find the image in already processed files
        img_processed = False
        if annotations:
            img_basename = os.path.splitext(os.path.basename(filename))[0]
            if df.index.str.startswith(img_basename).sum() > 0:
                img_processed = True

        if not img_processed or not annotations:
            # Read the image
            img = imread(filename)
            # Detect faces
            dets = face_detector(img, 1)

            # Iterate over all detected faces
            for i, d in enumerate(dets):
                count += 1
                # Predict landmark points
                shape = predictor(img, d)

                # Crop face
                face, points = crop_by_landmarks(img, shape_to_array(shape), pad=10)

                # Save face and landmarks
                base, ext = os.path.splitext(filename)
                face_filename = os.path.basename(base) + '_' + str(i) + ext
                imsave(os.path.join(path_out, face_filename), face)
                landmarks[face_filename] = array_to_str(points)

        bar.value += 1

    # Convert landmarks dictionary to DataFrame and add to csv
    landmarks_df = pd.DataFrame.from_dict(landmarks, columns=['points'], orient='index')
    landmarks_df.index.name = 'filename'
    landmarks_df['smile'] = 0
    landmarks_df['mouth_open'] = 0
    landmarks_df['labeled'] = False
    
    if annotations:
        df = df.append(landmarks_df, sort=False)
        df.to_csv(path_to_annotations)
    else:
        landmarks_df.to_csv(path_to_annotations)
    
    # Print statistics
    print('Found %d faces' % count)

    

class ImageList():
    '''
    Class instance reads images in the folder and iterates over them forward and backward,
    returning image in raw binary format.
    '''
    def __init__(self, folder, annotations, default_type='raw'):
        '''
        Initializer for the class instance.
        
        Parameters
        ----------
        folder : str
            Folder with images.
        annotations : str
            Path to csv file with annotations.
        default_type : str
            Default type for self.get_image() method.
            This method could return in raw binary format or as numpy array.
        '''
        # Get list of image pathes
        self.imgs = list(Path(folder).glob('*.jpg'))
        # Set current image in list
        if len(self.imgs) > 0:
            self.current = 0
        else:
            self.current = None    
        # Image count in list
        self.size = len(self.imgs)
        # Path to csv file
        self.path_to_df = annotations
        # Read csv to DataFrame
        self.df = load_annotations(annotations, filename_index=True)
        self.default_type = default_type
    
    
    def __getitem__(self, idx):
        '''
        Get image from list by index
        '''
        return self.imgs[idx]
    
    
    def get_image(self, as_type=None):
        '''
        Returns current file as raw binary or numpy array
        
        Parameters
        ----------
        as_type : str
            None - returns in defult format.
            'raw' - returns in binary format (for displaying in HTML widget).
            In other cases returns image as numpy array.
        
        Returns
        -------
        img : current image
        '''
        
        if as_type is None:
            as_type = self.default_type
        if as_type == 'raw':
            return self[self.current].read_bytes()
        else:
            return imread(self[self.current])
    
    
    def set_current(self, filename):
        '''
        Sets current image by filename.
        '''
        
        filenames = [x.name for x in self.imgs]
        self.current = filenames.index(filename)
    
    
    def filename(self):
        '''Returns filename of the current image.'''
        return self[self.current].name
    
    
    def get_labels(self):
        '''Return labels for the current image.'''
        return self.df.loc[self.filename(), ['smile', 'mouth_open']]
    
    
    def set_labels(self, labels):
        '''Set labels for the current image and saves changes to csv.'''
        self.df.loc[self.filename(), ['smile', 'mouth_open']] = [int(x) for x in labels]
        self.df.loc[self.filename(), ['labeled']] = True
        # Save changes
        self.df.to_csv(self.path_to_df)
    
    
    def get_landmarks(self):
        '''Return landmarks for the current image.'''
        points = self.df.loc[self.filename(), 'points']
        return str_to_array(points)
    
    
    def forward(self, as_type=None):
        '''Set current to the next image and return it in specified format.'''
        if as_type is None:
            as_type = self.default_type
        self.current = (self.current + 1) % self.size
        return self.get_image(as_type)
    
    
    def backward(self, as_type=None):
        '''Set current to the previous image and return it in specified format.'''
        if as_type is None:
            as_type = self.default_type
        self.current = (self.current - 1) % self.size
        return self.get_image(as_type)
    
    
    def delete(self, as_type=None):
        '''Delete current image, remove it from the list and DataFrame.'''
        if as_type is None:
            as_type = self.default_type
        # Delete image file
        os.remove(self[self.current])
        # Drop row from DataFrame
        self.df.drop(self.filename(), inplace=True)
        # Remove file from list of files
        self.imgs.pop(self.current)
        self.size -= 1
        # Update current image
        self.current = self.current % self.size
        return self.get_image(as_type)
    
    
    def viewer(self):
        '''View images with interactive viewer '''
        @interact(image=[x.name for x in self.imgs])
        def view_face(image):
            self.set_current(image)
            show_landmarks(self.get_image(as_type='numpy'), self.get_landmarks(), tuple(self.get_labels()))


def label_images(path_to_images, path_to_annotations):
    '''
    Provides user-interactive environment for labeling smiles and open mouth on images.
    Images have to be pre-processed with process_folder() function.
    
    Parameters
    ----------
    path_to_images : str
        Path to folder with images to label.
    path_to_annotations : str
        Path to annotaions csv file.
    '''
    
    # Create ImageList() object
    imgs = ImageList(path_to_images, path_to_annotations)
    # Get initial state for buttons
    labels = imgs.get_labels()
    
    # Create image and buttons widgets
    image_frame = widgets.Image(value=imgs.get_image(), width=400, height=400)
    title = widgets.Label(value=imgs.filename())
    button_next = widgets.Button(description='', tooltip='Next image', icon='forward')
    button_prev = widgets.Button(description='', tooltip='Previous image', icon='backward')
    button_save = widgets.Button(description='', tooltip='Save changes', icon='save')
    button_del = widgets.Button(description='', tooltip='Delete image', icon='trash')
    button_smile = widgets.ToggleButton(
        value=bool(labels['smile']), 
        description='Smile', 
        tooltip='Label smile', 
        icon='smile-o')
    button_mouth_open = widgets.ToggleButton(
        value=bool(labels['mouth_open']),
        description='Mouth open',
        tooltip='Label mouth open',
        icon='circle-o')
    
    # Distribute buttons in table
    column_1 = widgets.VBox([button_prev, widgets.Label()])
    column_2 = widgets.VBox([button_smile, button_save])
    column_3 = widgets.VBox([button_mouth_open, button_del])
    column_4 = widgets.VBox([button_next, widgets.Label()])
    buttons = widgets.HBox([column_1, column_2, column_3, column_4])

    def change_image(action):
        # Save current labels
        imgs.set_labels((button_smile.value, button_mouth_open.value))
        # Change current image
        image_frame.value = action()
        # Read labels
        labels = imgs.get_labels()
        button_smile.value, button_mouth_open.value = bool(labels['smile']), bool(labels['mouth_open'])
        # Add filename
        title.value=imgs.filename()        
    
    # Configure on_click events
    button_next.on_click(lambda x: change_image(imgs.forward))
    button_prev.on_click(lambda x: change_image(imgs.backward))
    button_del.on_click(lambda x: change_image(imgs.delete))
    button_save.on_click(lambda x: imgs.set_labels((button_smile.value, button_mouth_open.value)))
    
    # Display widgets
    display(buttons)
    display(image_frame)
    display(title)

    

def transfer_labels_from_folders(input_path, annotations):
    '''
    Update labels for already labeled images from folder with structure:
        --- input_path
            --- images
            --- smile
            --- open_mouth
    '''
    smile_path = os.path.join(input_path, 'smile')
    mouth_path = os.path.join(input_path, 'open_mouth')
    # Read filename in 'smile' folder
    smile = [os.path.splitext(img.name)[0] for img in Path(smile_path).glob('*.jpg')]
    # Read filename in 'open_mouth' folder
    mouth_open = [os.path.splitext(img.name)[0] for img in Path(mouth_path).glob('*.jpg')]
    # Open annotations DataFrame
    df = load_annotations(annotations, filename_index=True)

    # Update labels
    for img in smile:
        rows = df.index.str.startswith(img)
        df.loc[rows, 'smile'] = 1
    for img in mouth_open:
        rows = df.index.str.startswith(img)
        df.loc[rows, 'mouth_open'] = 1

    # Save changes
    df.to_csv(annotations)



def distribute_by_class(input_path, output_path, path_to_annotations, mode='binary'):
    '''
    Distribute images to folders with classnames.
    
    In binary mode folder structure would be:
        |ouput_path
        |---|smile
            |---|calm
            |---|smile
            |mouth
            |---|closed
            |---|open
    In categorical mode folder structure would be:
        |ouput_path
        |---|calm_closed
        |---|calm_open
        |---|smile_closed
        |---|smile_open
    
    Parameters
    ----------
    input_path : str
    output_path : str
    path_to_annotaions : 
    mode : str, 'binary' or 'cateorical'
    '''
    assert mode in ('binary', 'categorical')
    
    annotations = load_annotations(path_to_annotations, filename_index=True)
    
    if mode == 'binary':
        classes = classes_binary
        # Save stats for every class
        stats = {}
        for task in classes:
            stats[task] = {}
            for clss in classes[task]:
                stats[task][clss] = 0
        
        annotations.columns = ['smile', 'mouth', 'points', 'labeled']
        # Create class folders if needed
        for task in classes.keys():
            for clss in classes[task]:
                class_path = os.path.join(output_path, task, clss)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)

        # Get list of input files
        imgs = os.listdir(input_path)

        # Iterate over images in folder
        for filename in imgs:
            for task in classes.keys():
                try:
                    # Get class label
                    label = annotations.loc[filename, task]
                    # Copy to corresponding subfolder
                    shutil.copy2(os.path.join(input_path, filename), os.path.join(output_path, task, classes[task][label]))
                    # Update stats
                    stats[task][classes[task][label]] += 1
                except KeyError:
                    pass
        # Print stats
        print('Class statistics:')
        for task in classes.keys():
            print('{}:'.format(task))
            for clss in classes[task]:
                print('\t{}: {} images'.format(clss, stats[task][clss]))
        
    if mode == 'categorical':
        classes = classes_categorical
        # Save stats for every class
        stats = {}
        for clss in classes.values():
            stats[clss] = 0
        # Add categorical labels to annotations
        annotations['class'] = labels_to_categorical(annotations[['smile', 'mouth_open']])
        
        # Create class folders if needed
        for clss in classes.values():
            class_path = os.path.join(output_path, clss)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
        
        # Get list of input files
        imgs = os.listdir(input_path)
        
        for filename in imgs:
            try:
                # Get class label
                clss = annotations.loc[filename, 'class']
                class_name = classes[clss]
                # Copy to corresponding subfolder
                shutil.copy2(os.path.join(input_path, filename), os.path.join(output_path, class_name))
                # Update stats
                stats[class_name] += 1
            except KeyError:
                pass
        # Print stats
        print('Class statistics:')
        for clss in classes.values():
            print('{}: {} images'.format(clss, stats[clss]))


def main():
    pass

if __name__ == '__main__':
    main()
    