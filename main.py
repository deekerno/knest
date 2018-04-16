# UCF Senior Design 2017-18
# Group 38

from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from functools import partial
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from PIL import Image
import cv2
import gc
import math
import numpy as np
import os
import utils.blur as blur
import utils.compare as compare
import utils.global_var as gv
import utils.image_man as im
from kivy.config import Config

# remove os-provided border
Config.set('graphics', 'borderless', 'True')
# set window icon from default kivy image to knest logo
Config.set('kivy', 'window_icon', '/Users/ayylmao/Desktop/knest/assets/color_bird.png')

# all accepted images will be written to a subdirectory
# named 'processed'
DES_NAME = 'processed'
PATH_MAX = 8


def img_handler(img_path):
    """
    Determine whether or not a given file is an image
        img_path: (String) path to the file
    """
    try:
        # file is an image
        img = Image.open(img_path)
        img.close()
        return True

    except IOError:
        # file is not an image
        return False


class LandingScreen(Screen):
    """
    This is the splash screen; It displays the team name and logo for three
    seconds. User may move on to next screen by pressing on it
    """

    def __init__(self, **kwargs):
        """
        Display screen for three seconds
        """
        super(LandingScreen, self).__init__(**kwargs)
        Clock.schedule_once(self.switch, 3)

    def switch(self, dt):
        """
        Switch to folder selection screen
            dt: (int) time in seconds
        """
        self.manager.current = 'folder_select'


class FolderSelectScreen(Screen):
    """
    This screen is where users can select their input folder and toggle on/off
    the image comparison option
    """
    loadFile = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        """
        Create a pop-up inside the window to select a folder
        """
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select a Folder",
                            title_font='/Users/ayylmao/Desktop/knest/assets/Montserrat-Regular',
                            title_size='15sp',
                            content=content,
                            auto_dismiss=False,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        """
        Load the selected path
            path: (String) directory path chosen by user
        """
        self.update_path(gv.dir_paths)
        self.dismiss_popup()

    def add(self, path, popup_instance):
        # ensure that path has not already been selected
        if path not in gv.dir_paths:
            # ensure that we are able to write to the path
            if os.access(path, os.W_OK):
                gv.dir_paths.append(path)
                length = len(gv.dir_paths)

                # ensure that we are able to write to the path
                # and that we haven't reached the max amount of path
                if length <= PATH_MAX:
                    index = str(length - 1)
                    name = 'label' + index

                    # enable checkbox
                    popup_instance.ids[index].disabled = False
                    popup_instance.ids[index].active = True
                    # add path to path list in display
                    popup_instance.ids[name].text = os.path.normpath(os.path.basename(path))

                    # disable add button if max number of paths reached
                    if len(gv.dir_paths) == PATH_MAX:
                        popup_instance.ids.add.disabled = True

                # enable 'load' button
                popup_instance.ids.load.disabled = False

            # permission denied; display pop-up error message
            else:
                Factory.PermissionDenied().open()

    def remove(self, index, popup_instance):
        length = len(gv.dir_paths)

        # remove path from list of directory paths
        gv.dir_paths.pop(index)

        # move all paths below the one to remove up the checklist
        # if possible
        for i in range(index, length):
            if i == length - 1:
                # disabled checkbox
                popup_instance.ids[str(i)].active = False
                popup_instance.ids[str(i)].disabled = True

                # remove path at list index
                popup_instance.ids['label' + str(i)].text = ''
            else:
                # enable checkbox
                popup_instance.ids[str(i)].active = True
                popup_instance.ids[str(i)].disabled = False

                # set new path to next path
                popup_instance.ids['label' + str(i)].text = popup_instance.ids['label' + str(i + 1)].text

        # removing a path enables room to add a path,
        # so enable 'add' button
        popup_instance.ids.add.disabled = False

        # if there are no more paths listed, disable
        # 'load' button
        if len(gv.dir_paths) is 0:
            popup_instance.ids.load.disabled = True

    def update_path(self, dir_paths):
        """
        Display the selected path for user to see
            dir_path: (String) absolute path to user-selected folder
        """
        new_text = "Directory Name(s): "

        # only display relative path
        for i, path in enumerate(gv.dir_paths):
            new_text = new_text + os.path.normpath(os.path.basename(path))

            if i is not len(gv.dir_paths) - 1:
                new_text = new_text + ", "

        # update the path to show to user
        self.ids.path.text = new_text

    def check_path(self):
        """
        Check if the user has selected an input folder
        """
        if len(gv.dir_paths) is not 0:
            self.manager.current = 'black1'

        else:
            # if no path was given, prompt user for one
            self.ids.path.text = "No directory path given"

    def update_compare(self, popup_instance):
        """
        Update whether application will implement image comparison
        depending on state of toggle button
        """
        if popup_instance.ids.compare.active:
            gv.comp = True
        else:
            gv.comp = False

    def update_crop(self, popup_instance):
        """
        Update whether application will crop images depending on
        state of toggle button
        """
        if popup_instance.ids.crop.active:
            gv.crop = True
            # enable landscape text
            popup_instance.ids.caption3.opacity = 1
            # enable landscape switch
            popup_instance.ids.landscape.disabled = False
            # set landscape switch to on
            popup_instance.ids.landscape.active = True
            # set landscape global bool to True
            gv.landscape = True
        else:
            gv.crop = False
            # reduce opacity of landscape text
            popup_instance.ids.caption3.opacity = .5
            # set landscape switch to off
            popup_instance.ids.landscape.active = False
            # disable landscape switch
            popup_instance.ids.landscape.disabled = True
            # set landscape global bool to False
            gv.landscape = False

    def update_orientation(self, popup_instance):
        """
        Update whether application will crop images in landscape
        orientation depending on state of toggle button
        """
        if popup_instance.ids.landscape.active:
            gv.landscape = True
        else:
            gv.landscape = False


class BlackScreen1(Screen):
    """
    Transition screen
    """

    def switch(self, dt):
        """
        Switch to progress screen to begin application
            dt: (int) time in seconds
        """
        self.manager.current = 'progress'


class ProgressScreen(Screen):
    """
    This screen is where the classification model will be loaded. It
    also acts as a transition from folder selection to the beginning of
    the application process
    """

    def switch(self, dt):
        """
        Load the classification model and switch to next screen to begin
        processing images
            dt: (int) time in seconds
        """
        # make object detection import global
        global bf

        # load the model if it has not been done
        if not gv.load:
            import architectures.squeezenet.classifier as cl
            import utils.inference as bf

            # load the model
            gv.model = cl.ClassificationModel(
                (400, 400), 'output/squeezenet.tfl', 2)

            # instantiate object detection variables
            bf.instantiate()
            # update that model has been loaded
            gv.load = 1

        # determine how many files are in the path
        gv.num_files = len(os.listdir(gv.dir_paths[gv.path_index]))
        # switch to transition screen
        self.manager.current = 'black2'


class BlackScreen2(Screen):
    """
    Transition screen
    """

    def switch(self, dt):
        """
        Switch to process screen to begin processing images
            dt: (int) time in seconds
        """
        self.manager.current = 'process'


class ProcessScreen(Screen):
    """
    This screen is where all the images get processed. The steps are:
        1) blur detection
        2) bird classification
        3) bird localization

    The images and its results for each respective steps are displayed in
    real-time. Images are initially added to a global dictionary organized
    as {filename: numpy array} and get removed one-by-one every time an image
    fails a processing step
    """

    def update(self, dt):
        """
        Display processing results to screen in real-time for user to see
            dt: (int) time in seconds
        """
        # if blur detection has not been implemented
        if not gv.blur_step:
            # preventive measure: avoid out-of-index error
            if gv.index < gv.num_files:
                # ensure user did not alter working director mid-process
                if not (gv.num_files == len(os.listdir(gv.dir_paths[gv.path_index]))):
                    # display error message
                    Factory.OutOfIndex().open()
                    # reset the list of directory paths
                    # and the path index
                    gv.dir_paths = []
                    gv.path_index = 0
                    # reset all global variables
                    gv.reset()
                    # return to the folder selection screen having
                    # cancelled the process
                    self.manager.current = 'folder_select'
                    # unschedule the Clock.schedule_interval() method
                    return False

                # continue the process as usual
                else:
                    file_path = os.path.join(
                        gv.dir_paths[gv.path_index], os.listdir(gv.dir_paths[gv.path_index])[gv.index])

                    # avoid nonimages and hidden files
                    if img_handler(file_path):
                        # if this is the first pass into this step of the
                        # process, update the title of the process for
                        # user to see
                        if not gv.first_pass:
                            gv.first_pass = 1
                            # update stage of processing
                            self.ids.message.text = 'B L U R   D E T E C T I O N'
                            # remove image transparency
                            self.ids.image.color = (1, 1, 1, 1)

                        # display working image
                        self.ids.image.source = file_path
                        # reset result and add transparency back
                        self.ids.result.color = (0, 0, 0, 0)
                        self.ids.result.source = ''

                        # call blur detection on image
                        self.check_blur(file_path, os.listdir(
                            gv.dir_paths[gv.path_index])[gv.index])

                    # continue to next image
                    gv.index += 1

            # implemented blur detection on all images
            # update and move to next step
            else:
                gv.index = 0
                gv.first_pass = 0
                gv.files = list(gv.images.keys())
                gv.blur_step = 1

        # if bird classification has not been implemented
        # and if there are images to be processed
        elif not gv.bird_step and len(gv.images) is not 0:
            # preventive measure: avoid out-of-index error
            if gv.index < len(gv.files):
                file_path = os.path.join(gv.dir_paths[gv.path_index], gv.files[gv.index])

                # if this is the first pass into this step of the process,
                # update the title of the process for user to see
                if not gv.first_pass:
                    gv.first_pass = 1
                    # update stage of processing
                    self.ids.message.text = 'B I R D   C L A S S I F I C A T I O N'
                    # remove image transparency
                    self.ids.image.color = (1, 1, 1, 1)

                # display working image
                self.ids.image.source = file_path
                # reset result and add transparency back
                self.ids.result.color = (0, 0, 0, 0)
                self.ids.result.source = ''

                # call object classification on image
                self.check_class(
                    gv.images[gv.files[gv.index]], gv.files[gv.index])

                # continue to next image
                gv.index += 1

            # implemented bird classification on all images
            # update and move to next step
            else:
                gv.index = 0
                gv.first_pass = 0
                gv.files = list(gv.images.keys())
                gv.bird_step = 1

        # if bird localization has not been implemented
        # and if there are images to be processed
        elif not gv.birdbb_step and len(gv.images) is not 0:
            # preventive measure : avoid out-of-index error
            if gv.index < len(gv.files):
                file_path = os.path.join(gv.dir_paths[gv.path_index], gv.files[gv.index])

                # if this is the first pass into this step of the process,
                # update the title of the process for user to see and
                # change the format of the display
                if not gv.first_pass:
                    gv.first_pass = 1
                    # update stage of processing
                    self.ids.message.text = 'B I R D   L O C A L I Z A T I O N'
                    # add additional text descriptions
                    self.ids.previous.text = 'L A S T   P R O C E S S E D'
                    self.ids.current.text = 'C U R R E N T L Y   P R O C E S S I N G'
                    # remove image transparency
                    self.ids.image.color = (1, 1, 1, 1)
                    # change image and result position
                    self.ids.image.pos_hint = {
                        'center_x': 0.75, 'center_y': 0.5}
                    self.ids.result.pos_hint = {
                        'center_x': 0.25, 'center_y': 0.5}
                    # reset result and add transparency back
                    self.ids.result.color = (0, 0, 0, 0)
                    self.ids.result.source = ''

                # display working image
                self.ids.image.source = file_path
                # get working image's dimensions
                width = self.ids.image.width
                height = self.ids.image.height

                # preventive measure for proportional image scaling
                # for numpy array to texture conversion
                img_width = np.shape(gv.images[gv.files[gv.index]])[1]
                img_height = np.shape(gv.images[gv.files[gv.index]])[0]

                if img_height > img_width:
                    # calculate scaling
                    scale = height / img_height
                    width = img_width * scale

                # call object detection on image
                # a Clock event is scheduled to display images before
                # processing (required)
                Clock.schedule_once(partial(self.detect_bird, gv.images[
                    gv.files[gv.index]], gv.files[gv.index], width, height), 0)

                # continue to next image
                gv.index += 1

            # implemented bird/face detection on all images
            else:
                self.ids.image.opacity = 0
                gv.index = 0
                gv.files = list(gv.images.keys())
                gv.birdbb_step = 1

        # all images have been processed successfully
        # update and move to next screen
        else:
            # the folder path where all accepted images will be written
            gv.des_path = os.path.join(gv.dir_paths[gv.path_index], DES_NAME)

            # create the folder if it does not exist
            if not os.path.isdir(gv.des_path):
                os.makedirs(gv.des_path)

            # update new list of images
            gv.files = list(gv.images.keys())

            # switch to next screen after one second
            Clock.schedule_once(self.switch, 1)

            # update texture location to remove detection results later
            gv.canvas = self.ids.detection.canvas
            # collect any garbage not already gathered by python
            gc.collect()
            # unschedule kivy's Clock.schedule_interval() function
            return False

    def check_blur(self, img, filename):
        """
        Detect if an image is blurry and display results
            img: (ndarray) image file
            filename: (String) name of the image file
        """
        image, result = blur.detect_blur(img)

        # image is not blurry
        if result:
            # remove image transparency and display green check
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = '/Users/ayylmao/Desktop/knest/assets/yes.png'

            # add non-blurry image to image dictionary
            gv.images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image is blurry
        else:
            # remove image transparency and display red x
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = '/Users/ayylmao/Desktop/knest/assets/no.png'

    def check_class(self, img, filename):
        """
        Classify if an image contains a bird and display results
            img: (ndarray) image file
            filename: (String) name of the image file
        """
        resized_img = cv2.resize(img, (400, 400))
        prediction = gv.model.predict(resized_img)
        result = gv.model.classify(prediction)

        # image contains a bird
        if result:
            # remove image transparency and display green check
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = '/Users/ayylmao/Desktop/knest/assets/yes.png'

        # image does not contain a bird
        else:
            # remove image transparency and display red x
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = '/Users/ayylmao/Desktop/knest/assets/no.png'

            # remove image from dictionary
            gv.images.pop(filename)

    def detect_bird(self, img, filename, width, height, dt):
        """
        Localize bird(s) and bird face(s) in the image and display results
            img: (ndarray) image file
            filename: (String) name of the image file
            width: (float) width of img
            height: (float) height of img
            dt: (int) time in seconds
        """
        # run inference code
        image = bf.inference(filename, img)

        # clear any previous texture information as
        # to avoid continuously writing data on top of data
        self.ids.detection.canvas.clear()

        # create kivy texture from image ndarray
        texture = Texture.create(size=(width, height), colorfmt="rgb")
        # resize image for display
        image = cv2.resize(
            image, ((math.floor(width), math.floor(height))))
        # convert array to string
        data = image.tostring()
        # blit data to texture
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        # flip vertically to display upright
        texture.flip_vertical()

        # calculate position to center on left side of window
        x_pos = self.ids.image.parent.width * .25 - (width / 2)
        y_pos = self.ids.image.parent.height * .5 - (height / 2)

        # display result to screen
        with self.ids.detection.canvas:
            Rectangle(texture=texture, pos=(x_pos, y_pos),
                      size=(width, height))

        # no bird/faces were detected
        if not len(gv.boxes[filename]['birds']) or not len(gv.boxes[filename]['faces']):
            # remove image transparency and display red x
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = '/Users/ayylmao/Desktop/knest/assets/no.png'

            # remove image from dictionary
            gv.images.pop(filename)

        # bird face is detected
        else:
            # reset result and add transparency back in the
            # case that the previous image may not have had
            # a bird or face
            self.ids.result.color = (0, 0, 0, 0)
            self.ids.result.source = ''

    def switch(self, dt):
        """
        Switch to next appropriate screen
            dt: (int) time in seconds
        """
        if len(gv.images) == 0:
            if gv.path_index < len(gv.dir_paths) - 1:
                # continue to next path if there are more
                # directories to process
                gv.path_index += 1
                gv.reset()
                self.manager.current = 'black1'

            else:
                # if there are no images to write and no more directories
                # to process, switch to 'end' screen
                self.manager.current = 'black4'
                # reset the list of directory paths
                # and the path index
                gv.dir_paths = []
                gv.path_index = 0
                # reset all global variables for future passes
                gv.reset()
        else:
            # switch to transition screen
            self.manager.current = 'black3'


class BlackScreen3(Screen):
    """
    Transition screen
    """

    def switch(self, dt):
        """
        Switch to next screen based on corner toggle button
            dt: (int) time in seconds
        """
        if gv.comp:
            # switch to comparing screen
            self.manager.current = 'compare'
        else:
            # switch to writing screen
            self.manager.current = 'write'


class CompareScreen(Screen):
    """
    This screen is where the application reduces the number of similar images
    in the final subdirectory, based on user choice
    """

    def compare(self, dt):
        """
        Reduce number of similar images in a dictionary and update progress bar
            dt: (int) time in seconds
        """
        # set the progress bar maximum to the size of the dictionary
        length = self.ids.progress.max = len(gv.files)

        # preventive measure to avoid out-of-index error
        if gv.index < length:
            # set a comparison standard if there is none
            if gv.std == '':
                gv.std, gv.std_hash, gv.count = compare.set_standard(
                    gv.images, gv.files[gv.index])

            else:
                # compare the standard to the working image
                result = compare.limit(
                    gv.images[gv.files[gv.index]], gv.std_hash, gv.count)

                if result == 'update_std':
                    # non-similar image found; update standard
                    gv.std, gv.std_hash, gv.count = compare.set_standard(
                        gv.images, gv.files[gv.index])

                else:
                    if result == 'remove':
                        # too many similar images; remove from dictionary
                        gv.images.pop(gv.files[gv.index])

                    # continue with same standard
                    gv.count += 1

            # display progress to screen
            self.ids.loading.text = str(math.floor(
                ((gv.index + 1) / length) * 100)) + "%   C O M P L E T E"
            # update progress bar for user to see
            self.ids.progress.value = gv.index + 1
            # continue to next image
            gv.index += 1

        # compared entire dictionary; update and move on to next screen
        else:
            gv.index = 0
            # switch to writing screen after one second
            Clock.schedule_once(self.switch, 1)

            gv.files = list(gv.images.keys())
            # unschedule kivy's Clock.schedule_interval() function
            return False

    def switch(self, dt):
        """
        Switch to writing screen
            dt: (int) time in seconds
        """
        self.manager.current = 'write'


class WriteScreen(Screen):
    """
    This screen is where the accepted images get written into the final
    subdirectory, called 'processed'
    """

    def begin(self, dt):
        """
        Write final images to 'processed' folder and display progress for user
            dt: (int) time in seconds
        """

        # set the progress bar maximum to the size of the dictionary
        length = self.ids.progress.max = len(gv.images)

        # preventive measure to avoid out-of-index error
        if gv.index < length:
            # if cropping option is enabled
            if gv.crop:
                # call crop method on image to calculate bounding box
                # information and determine expansion and range of crop
                final_image, success = im.man(
                    gv.boxes[gv.files[gv.index]], gv.images[gv.files[gv.index]],
                    gv.landscape)
            # if user opts out of cropping
            else:
                final_image, success = gv.images[gv.files[gv.index]], 1

            if success:
                # preventive measure in the case that the subdirectory is
                # altered or removed during processing
                if not os.path.isdir(gv.des_path):
                    # display error message
                    Factory.NoDestination().open()
                    # reset the list of directory paths
                    # and the path index
                    gv.dir_paths = []
                    gv.path_index = 0
                    # reset all global variables
                    gv.reset()
                    # return to the folder selection screen having
                    # cancelled the process
                    self.manager.current = 'folder_select'
                    # unschedule the Clock.schedule_interval() method
                    return False

                # write accepted images to subdirectory
                self.write_to(gv.files[gv.index], final_image)

            # display progress to screen
            self.ids.loading.text = str(math.floor(
                ((gv.index + 1) / length) * 100)) + "%   C O M P L E T E"

            # update progress bar for user to see
            self.ids.progress.value = gv.index + 1
            # continue to next image
            gv.index += 1

        # all images have been written; update and move on to next screen
        # or folder, if applicable
        else:
            # reset all global variables for future passes
            gv.reset()

            # switch to end screen after one second
            Clock.schedule_once(self.switch, 1)

            # unschedule kivy's Clock.schedule_interval() function
            return False

    def write_to(self, filename, cropped_img):
        """
        Write image to 'processed' folder
            filename: (String): name of the image
            cropped_img: (ndarray) array representation of an image
        """
        img = Image.fromarray(cropped_img)
        img.save(os.path.join(gv.des_path, filename))

    def switch(self, dt):
        """
        Switch to writing screen
            dt: (int) time in seconds
        """
        # if there are more folders to process, move
        # to the processing screen
        if gv.path_index < len(gv.dir_paths) - 1:
            self.manager.current = 'black1'
            gv.path_index += 1

        else:
            # reset the list of directory paths
            # and the path index and move to 'end'
            # screen
            gv.dir_paths = []
            gv.path_index = 0
            self.manager.current = 'black4'


class BlackScreen4(Screen):
    """
    Transition screen
    """

    def switch(self, dt):
        """
        Switch to end screen
            dt: (int) time in seconds
        """
        self.manager.current = 'end'


class EndScreen(Screen):
    """
    This screen is where the user is notified that the process is complete
    """
    pass


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


# config.kv should not implement any screen manager stuff as it
# overrides any definitions in this file, and cause a lot of strife
Builder.load_file("/Users/ayylmao/Desktop/knest/assets/config.kv")

# Create the screen manager
sm = ScreenManager(transition=FadeTransition())
sm.add_widget(LandingScreen(name='landing'))
sm.add_widget(FolderSelectScreen(name='folder_select'))
sm.add_widget(BlackScreen1(name='black1'))
sm.add_widget(BlackScreen2(name='black2'))
sm.add_widget(ProgressScreen(name='progress'))
sm.add_widget(CompareScreen(name='compare'))
sm.add_widget(BlackScreen3(name='black3'))
sm.add_widget(WriteScreen(name='write'))
sm.add_widget(BlackScreen4(name='black4'))
sm.add_widget(ProcessScreen(name='process'))
sm.add_widget(EndScreen(name='end'))


class BirdApp(App):
    """
    Kivy module
    """

    def build(self):
        self.title = ''
        return sm


Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    BirdApp().run()
