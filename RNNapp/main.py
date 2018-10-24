import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.image import Image
from kivy.properties import ObjectProperty, StringProperty
from kivy.graphics import Rectangle
from kivy.graphics import Color
from kivy.graphics import Line
from keras.models import model_from_json
from kivy.clock import Clock
from PIL import Image as IMG
import numpy as np
from keras.models import load_model
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt


classes = [str(e) for e in range(0, 10)]


def load_image(infilename):
    img = IMG.open(infilename)
    img.load()
    img.thumbnail((28, 28), IMG.ANTIALIAS)
    data = np.asarray(img, dtype="int32")
    return data


class Dot(Widget):
    pass


class FullImage(Widget):

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            with self.canvas:
                touch.ud['line'] = Line(width=15, points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            if 'line' in touch.ud and touch.ud['line'] != 0:
                touch.ud['line'].points += [touch.x, touch.y]
            else:
                with self.canvas:
                    touch.ud['line'] = Line(width=15, points=(touch.x, touch.y))
        else:
            touch.ud['line'] = 0

    def on_touch_up(self, touch):
        self.export_to_png('drawnonimage.png')
        Clock.schedule_once(lambda dt: self.parent.predict_img())

    def clear_screen(self):
        self.canvas.clear()
        with self.canvas:
            Rectangle(source='blank.png')
            Color(0, 0, 0, 1)


class MainWidget(Widget):
    img = ObjectProperty(None)
    model = None
    prediction = StringProperty('')

    def setup_model(self):
        #self.model = load_model('model.h5')
        with open('model_architecture_1.json', 'r') as f:
            self.model = model_from_json(f.read())
            self.model.load_weights('model_weights_1.h5')

    def predict_img(self):
        whole_img = load_image('drawnonimage.png')
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.imshow(whole_img, cmap='gray')
        plt.show()
        alpha_img = whole_img[:, :, :1]
        alpha_img = alpha_img.reshape(1, 28, 28, 1) / 255
        prediction_class = (self.model.predict_classes(alpha_img, batch_size=128))[0]
        self.prediction = classes[prediction_class] + " - " + "{0:.2f}".format((self.model.predict_proba(alpha_img, batch_size=128))[0][prediction_class] * 100) + "%"

    def clear_screen(self):
        self.remove_widget(self.img)
        self.img = FullImage()
        self.img.size = (450, 450)
        self.img.center_x = self.center_x
        self.img.center_y = self.center_y + 50
        self.add_widget(self.img)


class ReadingNeuralNetworkApp(App):
    def build(self):
        mainWindow = MainWidget()
        Clock.schedule_once(lambda dt: mainWindow.setup_model())
        return mainWindow


if __name__ == '__main__':
    classes.extend([str(chr(e)) for e in range(65, 91)])
    classes.extend(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])
    ReadingNeuralNetworkApp().run()
