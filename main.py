# UCF Senior Design 2017-18
# Group 38

from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from PIL import Image
import blur
import os

# global variables
dir_path = ""
num_files = 0
index = 0


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
        self.manager.current = 'black2'


class BlackScreen2(Screen):

    def switch(self, dt):
        self.manager.current = 'process'


class ProcessScreen(Screen):

    def update(self, dt):
        # global references
        global dir_path, index, num_files

        # preventive measure: avoid out of index error
        if index < num_files:
            file_path = os.path.join(dir_path, os.listdir(dir_path)[index])

            # avoid nonimages and hidden files
            if img_handler(file_path):
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
                blur_result = self.check_blur(file_path)

                # if blur detection produces a result,
                # move on to next image by updating number
                if blur_result is True or blur_result is False:
                    index += 1

            # indicates a nonimage or hidden file; move on to next file
            else:
                index += 1

        # reached end of directory; reset all global variables and change
        # screens
        else:
            index = 0
            num_files = 0
            dir_path = ""
            self.manager.current = 'black3'
            # unschedule kivy's Clock.schedule_interval() function
            return False

    # call blur detection and display results
    def check_blur(self, img):
        # if image is not blurry, display green checkmark
        if blur.check_sharpness(img, 100):
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/yes.png'
            return True
        # otherwise, display red x
        else:
            self.ids.result.color = (1, 1, 1, 1)
            self.ids.result.source = 'assets/no.png'
            return False
        # preventive measure: will never actually reach here
        return None


class BlackScreen3(Screen):

    def switch(self, dt):
        self.manager.current = 'end'


class EndScreen(Screen):
    print("EndScreen")


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
sm.add_widget(ProcessScreen(name='process'))
sm.add_widget(EndScreen(name='end'))


# Pass it onto the kivy module
class BirdApp(App):

    def build(self):
        return sm


Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    BirdApp().run()
