# UCF Senior Design 2017-18
# Group 38

from kivy.app import App
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
import os


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
        numFiles = 0

        for i in os.listdir(path):
            if not i.startswith('.'):
                numFiles += 1
                print(i)

        print("number of files: ", numFiles)

        dir_path = path

        self.update_path(dir_path)

        self.dismiss_popup()

    def update_path(self, dir_path):
        new_text = "Path: " + dir_path
        self.ids.path.text = new_text

    # All classes must be passed at the end so Kivy can interpret them
    pass


class ProgressScreen(Screen):
    pass


class EndScreen(Screen):
    pass


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


# Create the screen manager and add the screens to it
sm = ScreenManager()
sm.add_widget(FolderSelectScreen(name='folder_select'))
sm.add_widget(ProgressScreen(name='progress'))
sm.add_widget(EndScreen(name='end'))


# Pass it onto the kivy module
class BirdApp(App):
    pass


Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    BirdApp().run()
