# UCF Senior Design 2017-18
# Group 38

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.animation import Animation
import os

# global variable
dir_path = ""
num_files = 0

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
        num_files = len([f for f in os.listdir(dir_path) if not f.startswith('.')])

        self.update_path(dir_path)

        self.dismiss_popup()

    def update_path(self, dir_path):
        new_text = "Directory Path: " + dir_path
        self.ids.path.text = new_text

    def check_path(self):
        if not dir_path == "":
            self.manager.current = 'progress'
        else:
            self.ids.path.text = "No directory path given"


class LandingScreen(Screen):

    def __init__(self, **kwargs):
        super(LandingScreen, self).__init__(**kwargs)
        Clock.schedule_once(self.switch, 3)

    def switch(self, dt):
        self.manager.current = 'folder_select'


class ProgressScreen(Screen):
    print("ProgressScreen")

    # def display_images(self):
    #     for filename in os.listdir(dir_path):
    #         if not filename.startswith('.'):
    #             self.ids.image.source = filename



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
sm.add_widget(ProgressScreen(name='progress'))
sm.add_widget(EndScreen(name='end'))


# Pass it onto the kivy module
class BirdApp(App):

    def build(self):
        return sm


Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    BirdApp().run()
