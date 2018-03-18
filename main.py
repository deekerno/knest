# UCF Senior Design 2017-18
# Group 38

import cv2
import gc
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from PIL import Image
import os
import utils.blur as blur
import utils.compare as compare
# import architectures.buff_bobo.classifier as cl

DES_NAME = 'processed'

# global variables for entire process
blur_step = 0
bird_step = 0
load = 0
model = None

# global variables to read images in user-given path
first_pass = 0
dir_path = ""
des_path = ""
num_files = 0
index = 0

# global variables to keep track of image data
images = {}

# global variables for image comparison
comp = 0
std = ''
std_hash = None
count = 0

# global variables for writing to subdirectory
files = []


def img_handler(img_path):
    try:
        # file is an image
        img = Image.open(img_path)
        img.close()
        return True

    except IOError:
        # file is not an image
        return False

# Define behavior specific to a particular screen
class FolderSelectScreen(Screen):
    loadFile = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    # Create a pop-up inside the window to select a folder
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select folder", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    # Store the path in a variable to send to the backend
    def load(self, path, filename):
        global dir_path, num_files

        dir_path = path
        num_files = len(os.listdir(dir_path))

        self.update_path(dir_path)

        self.dismiss_popup()

    def update_path(self, dir_path):
        # only display relative path
        new_text = "Directory Name: " + \
            os.path.normpath(os.path.basename(dir_path))
        self.ids.path.text = new_text

    def check_path(self):
        if not dir_path == "":
            self.manager.current = 'black1'
        else:
            self.ids.path.text = "No directory path given"

    def update_toggle(self):
        global comp

        # depending on status of switch, update whether
        # application will implement image comparison
        if self.ids.choice.active:
            comp = 1
        else:
            comp = 0


class LandingScreen(Screen):

    def __init__(self, **kwargs):
        super(LandingScreen, self).__init__(**kwargs)
        Clock.schedule_once(self.switch, 3)

    def switch(self, dt):
        self.manager.current = 'folder_select'


class BlackScreen1(Screen):

    def switch(self, dt):
        self.manager.current = 'progress'


class ProgressScreen(Screen):

    def switch(self, dt):
        global load, model
        if not load:
            print("Loading model")

            import architectures.buff_bobo.classifier as cl
            model = cl.ClassificationModel((112, 112), 'output/buff_bobo-670')

            print("Done loading model")
            load = 1

        self.manager.current = 'black2'


class BlackScreen2(Screen):

    def switch(self, dt):
        self.manager.current = 'process'


class ProcessScreen(Screen):

    def update(self, dt):
        global blur_step, bird_step
        global first_pass, dir_path, index, num_files, comp, images, des_path
        global files, load, model

        if not blur_step:
            print("implementing blur detection")
            # preventive measure: avoid out of index error
            if index < num_files:
                file_path = os.path.join(dir_path, os.listdir(dir_path)[index])

                # avoid nonimages and hidden files
                if img_handler(file_path):
                    if not first_pass:
                        first_pass = 1
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
                    blur_result = self.check_blur(file_path, os.listdir(dir_path)[index])

                    # if blur detection produces a result,
                    # move on to next image by updating number
                    if blur_result is True or blur_result is False:
                        index += 1

                # indicates a nonimage or hidden file; move on to next file
                else:
                    index += 1

            else:
                self.ids.message.text = ''

                self.ids.image.color = (0, 0, 0, 0)
                self.ids.image.source = ''

                self.ids.result.color = (0, 0, 0, 0)
                self.ids.result.source = ''

                index = 0
                first_pass = 0
                files = list(images.keys())
                blur_step = 1

        elif not bird_step:
            if index < len(files):
                file_path = os.path.join(dir_path, files[index])

                print("implementing bird classification")
                if not first_pass:
                    first_pass = 1
                    self.ids.message.text = 'B I R D   C L A S S I F I C A T I O N'

                    # remove image transparency
                    self.ids.image.color = (1, 1, 1, 1)

                # display working image
                self.ids.image.source = file_path
                # reset result and add transparency back
                self.ids.result.color = (0, 0, 0, 0)
                self.ids.result.source = ''

                # call object classification on image
                class_result = self.check_class(images[files[index]], files[index])

                if class_result is True or class_result is False:
                    index += 1

            else:
                index = 0
                files = list(images.keys())
                bird_step = 1

        # reached end of directory, begin writing to subdirectory
        # switch to writing screen
        else:
            index = 0
            # the folder we will write all accepted images to
            des_path = os.path.join(dir_path, DES_NAME)

            # if it doesn't exist, make the folder
            if not os.path.isdir(des_path):
                os.makedirs(des_path)

            files = list(images.keys())

            # switch to writing/compare screen
            self.manager.current = 'black3'

            # collect any garbage not already gathered by python
            gc.collect()

            # unschedule kivy's Clock.schedule_interval() function
            return False

    # call blur detection and display results
    def check_blur(self, img, filename):
        global images

        image, result = blur.detect_blur(img)

        # if image is not blurry, display green checkmark
        if result:
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/yes.png'

            # images that pass blur detection will be added to
            # image dictionary
            images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return True
        # otherwise, display red x
        else:
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/no.png'
            return False
        # preventive measure: will never actually reach here
        return None

    def check_class(self, img, filename):
        resized_img = cv2.resize(img, (112, 112))
        prediction = model.predict([resized_img])
        result = model.classify(prediction)

        # if image has a bird, display green checkmark
        if result:
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/yes.png'
            return True
        # otherwise, display red x
        else:
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/no.png'
            images.pop(filename)
            return False
        # preventive measure: will never actually reach here
        return None


class CompareScreen(Screen):
    
    def compare(self, dt):
        global images, std, count, index, std_hash, files

        length = self.ids.progress.max = len(files)

        if index < length:
            self.ids.loading.text = "Loading " + str(index + 1) + " of " + str(length)
            if std == '':
                std, std_hash, count = compare.set_standard(images, files[index])

            else:
                result = compare.limit(images[files[index]], std_hash, count)

                if result == 'remove':
                    images.pop(files[index])
                    count += 1
                elif result == 'continue':
                    count += 1
                elif result == 'update_std':
                    std, std_hash, count = compare.set_standard(images, files[index])

            self.ids.progress.value = index + 1
            index += 1

        else:
            index = 0
            self.manager.current = 'write'
            files = list(images.keys())
            return False


class BlackScreen3(Screen):
    
    def switch(self, dt):
        if comp:
            self.manager.current = 'compare'
        else:
            self.manager.current = 'write'


class WriteScreen(Screen):

    def begin(self, dt):
        global files, index

        length = self.ids.progress.max = len(images)

        if index < length:
            self.ids.loading.text = "Loading " + str(index + 1) + " of " + str(length)
            # write accepted images to subdirectory
            self.write_to(files[index])
            self.ids.progress.value = index + 1

            index += 1

        else:
            # reset all global variables for future passes
            self.reset()

            # switch to end screen
            self.manager.current = 'black4'

            return False

    def write_to(self, filename):

        img = Image.fromarray(images[filename])
        img.save(os.path.join(des_path, filename))

    def reset(self):
        global first_pass, index, num_files, dir_path, des_path, images
        global std, count, files, std_hash, blur_step, bird_step

        # reset all global variables for use in future passes
        first_pass = 0
        index = 0
        num_files = 0
        dir_path = ""
        des_path = ""
        std = ''
        count = 0
        files = []
        std_hash = None

        blur_step = 0
        bird_step = 0

        # empty images dictionary
        images.clear()


class BlackScreen4(Screen):

    def switch(self, dt):
        self.manager.current = 'end'


class EndScreen(Screen):
    pass


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


# config.kv should not implement any screen manager stuff as it
# overrides any definitions in this file, and cause a lot of strife
Builder.load_file("config.kv")

# Create the screen manager
sm = ScreenManager(transition=FadeTransition())
sm.add_widget(LandingScreen(name='landing'))
sm.add_widget(FolderSelectScreen(name='folder_select'))
sm.add_widget(BlackScreen1(name='black1'))
sm.add_widget(BlackScreen2(name='black2'))
sm.add_widget(ProgressScreen(name='progress'))
sm.add_widget(BlackScreen3(name='black3'))
sm.add_widget(CompareScreen(name='compare'))
sm.add_widget(WriteScreen(name='write'))
sm.add_widget(BlackScreen4(name='black4'))
sm.add_widget(ProcessScreen(name='process'))
sm.add_widget(EndScreen(name='end'))


# Pass it onto the kivy module
class BirdApp(App):

    def build(self):
        self.title = ''
        # removes os-created window border
        # without this, we need to create custom exit and
        # minimize buttons
        Window.borderless = True
        return sm


Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    BirdApp().run()
